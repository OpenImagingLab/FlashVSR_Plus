"""Opt-in TC4 full-clip transport path; model weights and attention are unchanged."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from inference.output import IncrementalVideoWriter, mux_source_audio
from inference.postprocessing.color_correction import WaveletColorCorrector
from inference.preprocessing.resident import prepare_rgb8
from inference.randomness import set_inference_seed
from inference.request import InferenceRequest


@dataclass
class FastInferenceResult:
    """RGB8 CPU frames can feed another stage without intermediate MP4 encoding."""
    frames_rgb8: np.ndarray
    output_path: str | None
    frames: int
    fps: float
    metadata: dict


def _read_rgb8(request):
    capture = cv2.VideoCapture(str(Path(request.input_path).expanduser()))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open input video: {request.input_path}")
        fps = float(request.fps or capture.get(cv2.CAP_PROP_FPS) or 24)
        while request.max_frames is None or len(frames) < request.max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from: {request.input_path}")
    return np.stack(frames), fps


def _fast_device(config):
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("fast buffers require a CUDA device")
    return device


@torch.inference_mode()
def infer_fast(pipeline, request: InferenceRequest, *, rgb8=None,
               tensor_resize=False, triton_wavelet=False,
               resident_budget_gib=5.0, progress=None, video_preset="veryfast",
               delivery_size=None):
    """Run TC4/tiny/auto using one resident BF16 conditioning allocation.

    Supply decoded [F,H,W,3] uint8 ``rgb8`` to skip source decoding. The request
    input_path is then only used for optional audio muxing; omit output_path to
    pass ``result.frames_rgb8`` directly to a denoiser or final encoder.
    No VRAM/KV policy is silently changed; budget both alongside this allocation.
    """
    request.validate()
    config = pipeline.config
    if (config.temporal_compression != 4 or config.decoder.kind != "tiny"
            or config.decoder.decode_every != "auto"):
        raise ValueError("fast buffers require TC4, the tiny decoder and decode-every=auto")
    if request.lq_video is not None:
        raise ValueError("Use rgb8= for decoded frames; normalized lq_video is not an RGB8 input")
    if request.height is None or request.width is None:
        raise ValueError("fast buffers require explicit height and width")
    if request.height % 128 or request.width % 128:
        raise ValueError("height and width must be divisible by 128")
    if resident_budget_gib <= 0 or not np.isfinite(resident_budget_gib):
        raise ValueError("resident_budget_gib must be positive and finite")
    device = _fast_device(config)
    if config.dtype != "bf16":
        raise ValueError("The measured fast buffers path requires bf16")
    set_inference_seed(config.seed)
    times = {}
    start = time.perf_counter()
    if rgb8 is None:
        rgb8, fps = _read_rgb8(request)
    else:
        fps = float(request.fps or 24)
        if request.max_frames is not None:
            rgb8 = rgb8[:request.max_frames]
    times["source_decode_seconds"] = time.perf_counter() - start

    def checkpoint(label, start):
        torch.cuda.synchronize(device)
        times[label] = time.perf_counter() - start

    start = time.perf_counter()
    resident = prepare_rgb8(
        rgb8, request.height, request.width, device=device,
        tensor_resize=tensor_resize,
        max_resident_bytes=int(resident_budget_gib * 1024**3),
    )
    checkpoint("input_preparation_seconds", start)
    count = len(resident.reference_rgb8)
    effective = ((count - 1 + 3) // 4) * 4 + 1
    if progress:
        progress("Resident input ready")
    start = time.perf_counter()
    output = pipeline.runner.run(
        resident.padded[:, :, :effective], request.context,
        padded_lq=resident.padded, progress=progress,
    )
    checkpoint("runner_seconds", start)
    generated = output.video[:, :, :count]
    reference = resident.reference_rgb8
    resize_control = resident.resize_control
    del resident
    # The decoder returns CPU BF16. Free the input's GPU allocation before
    # processing colour in bounded chunks; retain only the CPU RGB8 reference.
    start = time.perf_counter()
    pixels = np.empty((count, request.height, request.width, 3), dtype=np.uint8)
    corrector = WaveletColorCorrector(
        levels=config.color_correction.levels, chunk_size=4,
        backend="triton" if triton_wavelet else "torch",
    ) if config.color_correction.enabled else None
    if triton_wavelet and corrector is not None:
        from inference.postprocessing.wavelet_cuda import validate_wavelet
        validate_wavelet(device)
    with torch.backends.cudnn.flags(allow_tf32=False):
        for offset in range(0, count, 4):
            stop = min(count, offset + 4)
            value = generated[:, :, offset:stop].to(device, non_blocking=True)
            if corrector is not None:
                # Transfer bytes before casting, keeping FP32 conversion on CUDA.
                style = torch.from_numpy(np.ascontiguousarray(reference[offset:stop]))
                style = style.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
                style.div_(127.5).sub_(1)
                value = corrector(value, style.permute(1, 0, 2, 3).unsqueeze(0))
                del style
            # Match the existing RGB8 rounding, transferring compact output once.
            rgb = value[0].permute(1, 2, 3, 0).float().add_(1).mul_(.5)
            rgb.clamp_(0, 1).mul_(255).round_()
            pixels[offset:stop] = rgb.byte().contiguous().cpu().numpy()
            del value, rgb
    checkpoint("colour_and_rgb8_seconds", start)
    metadata = {
        **output.metadata, "fast_buffers": True, "frames": count,
        "tensor_resize": tensor_resize, "resize_control": resize_control,
        "triton_wavelet": triton_wavelet and corrector is not None,
        "timings": times,
    }
    destination = None
    if request.output_path is not None:
        destination = Path(request.output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A separate video-only file keeps muxing safe when source audio exists.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="flashvsr-", dir=destination.parent) as temporary:
            encoded = Path(temporary) / "video.mp4"
            start = time.perf_counter()
            writer = IncrementalVideoWriter(encoded, fps, preset=video_preset,
                                            output_size=delivery_size)
            try:
                writer.append_rgb8(pixels)
            finally:
                writer.close()
            metadata["artifact"] = writer.verify()
            mux_source_audio(encoded, request.input_path, destination)
            metadata["artifact"]["path"] = str(destination)
            times["mp4_export_and_mux_seconds"] = time.perf_counter() - start
        if progress:
            progress("Video saved")
    return FastInferenceResult(pixels, str(destination) if destination else None,
                               count, fps, metadata)
