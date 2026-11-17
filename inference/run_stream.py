# Modified: optional resident-input inference and reduced frame-transfer/copy overhead.
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from inference import (
    InferenceRequest,
    build_inference_pipeline,
)
from inference.console import print_banner, print_stage
from inference.presets import INFERENCE_MODES, model_paths_for_mode, stream_config_for_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a FlashVSR+ stream inference checkpoint"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-every", default="auto",
                        help="auto loads the full input; a positive multiple of 4 enables incremental reading and decoding")
    parser.add_argument(
        "--hardware-profile",
        choices=["auto", "generic"],
        default="auto",
        help="backend policy: auto detects installed GPU kernels; generic uses SDPA",
    )
    parser.add_argument(
        "--attn-kernel",
        choices=["auto", "sdpa", "block_sparse", "triton_block_sparse"],
        default="auto",
        help="override the causal/sparse stream attention backend",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=INFERENCE_MODES,
        default="turbo",
        help="inference preset: turbo uses hybrid-b25; base and v1 disable hybrid routing",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="override the mode checkpoint directory containing dit.safetensors and lq_proj_in.safetensors",
    )
    parser.add_argument(
        "--serial-tiny-decoder",
        action="store_true",
        help="force strict serial TinyDecoder execution",
    )
    parser.add_argument(
        "--tiny-decoder-mode",
        choices=["auto", "serial", "parallel"],
        default="auto",
        help="TinyDecoder mode: auto, serial, or full temporal parallel",
    )
    parser.add_argument(
        "--tiny-decoder-channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use channels-last BF16 TinyDecoder convolutions (default: true)",
    )
    parser.add_argument(
        "--cf",
        "--color-correction",
        dest="color_correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable low-frequency color correction (default: disabled)",
    )
    parser.add_argument(
        "--color-correction-levels",
        type=int,
        default=5,
        help="wavelet color-correction levels (default: 5)",
    )
    parser.add_argument(
        "--color-correction-chunk-size",
        type=int,
        default=16,
        help="frames processed per color-correction chunk (default: 16)",
    )
    parser.add_argument(
        "--offload-original-cache",
        action="store_true",
        help="keep original-attention layer KV caches on CPU between stream chunks",
    )
    parser.add_argument(
        "--cache-offload-mode",
        choices=["auto", "never", "always"],
        default="auto",
        help="stream KV policy: auto by resolution/VRAM, or force never/always",
    )
    parser.add_argument(
        "--temporal-compression",
        "--temporal_compression",
        type=int,
        choices=[1, 2, 4],
        default=4,
        help="effective temporal compression: 4 is default, 1 disables temporal compression",
    )
    parser.add_argument(
        "--temporal-merge",
        "--temporal_merge",
        choices=["gaussian", "first"],
        default="gaussian",
        help="how repeated decoded frames are collapsed when temporal compression is 1 or 2",
    )
    parser.add_argument(
        "--temporal-gaussian-sigma",
        "--temporal_gaussian_sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used by --temporal-merge gaussian",
    )
    parser.add_argument(
        "--video-preset",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
        ],
        default="veryfast",
        help="x264 encoding preset for output video speed/size tradeoff",
    )
    parser.add_argument("--fast-buffers", action="store_true",
                        help="opt in to resident TC4 BF16 input and bounded CUDA colour/RGB8 output")
    parser.add_argument("--tensor-resize", action="store_true",
                        help="opt-in two-pass CUDA bicubic approximation to PIL; requires --fast-buffers")
    parser.add_argument("--triton-wavelet", action="store_true",
                        help="use checked FP32 Triton wavelet blur; requires --fast-buffers")
    parser.add_argument("--resident-budget-gib", type=float, default=5.0,
                        help="maximum resident input allocation; excludes models, KV and workspaces")
    parser.add_argument("--cache-offload-layers", type=int, nargs="+", default=None,
                        help="explicit exact KV layers to keep in RAM, e.g. 0 through 11 for benchmark Base")
    parser.add_argument("--delivery-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
                        help="optional final CPU encoder size; model canvas remains unchanged")
    parser.add_argument("--timings-json", default=None,
                        help="save fast-path stage timings and metadata as JSON")
    args = parser.parse_args()
    if (args.tensor_resize or args.triton_wavelet or args.timings_json or args.delivery_size) and not args.fast_buffers:
        parser.error("--tensor-resize, --triton-wavelet, --delivery-size and --timings-json require --fast-buffers")
    if args.cache_offload_layers is not None and any(x < 0 or x >= 30 for x in args.cache_offload_layers):
        parser.error("--cache-offload-layers must be between 0 and 29")
    if args.cache_offload_layers and args.cache_offload_mode == "never":
        parser.error("explicit offload layers conflict with --cache-offload-mode never")
    return args


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tiny_decoder_mode = (
        "serial" if args.serial_tiny_decoder else args.tiny_decoder_mode
    )
    config = stream_config_for_mode(
        args.mode,
        device=args.device,
        seed=args.seed,
        decode_every=args.decode_every,
        tiny_decoder_mode=tiny_decoder_mode,
        tiny_decoder_channels_last=args.tiny_decoder_channels_last,
        offload_original_cache=args.offload_original_cache,
        cache_offload_mode=args.cache_offload_mode,
        temporal_compression=args.temporal_compression,
        temporal_merge=args.temporal_merge,
        temporal_gaussian_sigma=args.temporal_gaussian_sigma,
        attn_kernel=args.attn_kernel,
        hardware_profile=args.hardware_profile,
        color_correction=args.color_correction,
        color_correction_levels=args.color_correction_levels,
        color_correction_chunk_size=args.color_correction_chunk_size,
    )
    if args.cache_offload_layers is not None:
        config.stream_cache_offload_layers = tuple(args.cache_offload_layers)
    if args.fast_buffers and (config.temporal_compression != 4
                             or config.decoder.decode_every != "auto"):
        raise ValueError("--fast-buffers requires TC4 and --decode-every auto")
    paths = model_paths_for_mode(root, args.mode, args.ckpt_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this command with host GPU access")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    print_banner()
    print_stage("Loading model weights...")
    pipeline = build_inference_pipeline(config, paths)
    torch.cuda.synchronize(device)
    print_stage("Model loading complete", done=True)
    print_stage(
        "Attention mode: TURBO / HYBRID-B25"
        if config.is_hybrid_sliding
        else "Attention mode: BASE"
    )
    print_stage(
        f"Configuration: {args.mode} · {config.attn_kernel} · "
        f"{config.dtype} · {config.device} · decoder={config.decoder.kind}"
    )
    torch.cuda.reset_peak_memory_stats(device)
    print_stage("Running inference...")
    infer = pipeline.infer
    extra = {}
    if args.fast_buffers:
        from inference.fast import infer_fast
        infer = lambda request, **kwargs: infer_fast(pipeline, request, **kwargs)
        extra = dict(tensor_resize=args.tensor_resize,
                     triton_wavelet=args.triton_wavelet,
                     resident_budget_gib=args.resident_budget_gib,
                     delivery_size=tuple(reversed(args.delivery_size)) if args.delivery_size else None)
    result = infer(
        InferenceRequest(
            input_path=args.input,
            output_path=output,
            height=args.height,
            width=args.width,
            max_frames=args.max_frames,
        ),
        progress=lambda message: print_stage(message, done=True),
        video_preset=args.video_preset,
        **extra,
    )
    if args.timings_json:
        import json
        timings_path = Path(args.timings_json).expanduser()
        timings_path.parent.mkdir(parents=True, exist_ok=True)
        timings_path.write_text(json.dumps(result.metadata, indent=2) + "\n")
    print_stage("Inference complete", done=True)
    torch.cuda.synchronize(device)
    print_stage(f"Saved to {result.output_path}", done=True)


if __name__ == "__main__":
    main()
