<!-- language: en -->
# Resident-buffer inference

The `--fast-buffers` path ports measured full-clip transport optimizations to
the official CLI. It requires BF16, temporal compression 4, the TinyDecoder and
`--decode-every auto`. The normal CLI input path remains available by omitting
the flag. Decoder output assembly is optimized for both paths.

Install the existing repository dependencies and model weights as described in
the main README. No additional mandatory Python dependency is introduced.

## RTX 5090 example: 144 frames, 768p to 2K

The measured source was 768×1344, 144 frames at 24 fps (six seconds).
Inference uses the divisible-by-128 canvas 1536×2688. The final CPU encoder
resizes to delivery dimensions 1440×2560 without another intermediate MP4.
Dimensions below are height then width.

```bash
python -m inference.run_stream \
  --mode turbo --input input-768p.mp4 --output turbo-2k.mp4 \
  --height 2688 --width 1536 --max-frames 144 \
  --attn-kernel triton_block_sparse --serial-tiny-decoder \
  --cache-offload-mode never \
  --fast-buffers --tensor-resize --triton-wavelet --cf \
  --delivery-size 2560 1440 --timings-json turbo-timings.json
```

For the measured Base configuration, replace `--mode turbo` with `--mode base`
and replace `--cache-offload-mode never` with:

```bash
--cache-offload-mode auto --cache-offload-layers 0 1 2 3 4 5 6 7 8 9 10 11
```

Those are the tested residency choices for this 32 GB GPU and geometry, not a
universal automatic VRAM policy. The full resident input takes about 3.7 GiB
in addition to model weights, KV and workspaces. `--resident-budget-gib` only
limits that input allocation. For longer clips or lower VRAM, use the ordinary
numeric `--decode-every` path rather than enabling the full-clip resident path.
No attention history is shortened and no model quantization is added.

`--tensor-resize` is optional. Omit it to retain PIL bicubic. The two-pass CUDA
bicubic approximation checks the first source batch against PIL and rejects
differences above 2/255 maximum or 0.3/255 mean. This is a guard on that batch,
not a guarantee for every pixel of the entire video. On checked output frames,
the benchmark renders exceeded 51 dB PSNR against the PIL control; this is
numerical consistency, not a ground-truth quality score.

`--triton-wavelet` is optional and only applies when colour correction is on.
It checks FP32 blur radii 1, 2, 4, 8 and 16 against the stock implementation on
first use. Omit it to keep the PyTorch wavelet implementation. Spatial tiling,
attention, model weights, temporal history and serial decoder execution remain
unchanged in the measured settings.

## Connected pipelines

```python
from inference import InferenceRequest
from inference.fast import infer_fast

# Build and retain `pipeline` using the normal presets/model loader.
result = infer_fast(
    pipeline,
    InferenceRequest(input_path="source.mp4", height=2688, width=1536, fps=24),
    rgb8=decoded_rgb8_frames,  # [F,H,W,3] CPU numpy uint8; no source decode
    tensor_resize=True,
    triton_wavelet=True,
)
frames_for_denoiser = result.frames_rgb8
```

Omitting `output_path` skips encoding. The return type is `FastInferenceResult`
with explicit RGB8 CPU frames, not the normal pipeline's normalized tensor.
The input path is only read for video decoding when `rgb8` is omitted, or for
audio muxing when saving an MP4. RGB8 output retains the model canvas; optional
delivery resizing happens only in the final encoder.

## Benchmark and validation boundaries

The original GPU benchmark implementation measured Turbo at 53.05–58.48 seconds
(three warm runs) and Base at 68.02–77.73 seconds (two warm runs), excluding
model loading, source MP4 decoding and final MP4 export. Colour processing,
RGB8 transfer and wrapper cleanup/validation were retained. Hardware: RTX 5090,
two CPU cores, at most 64 GiB RAM. Separate final exports took about 28–29 seconds.
The CLI refactor in this archive has CPU regression and encoder tests; it has
not received a new full-model GPU timing run. Historical timings are not a
guarantee for every installation.

```bash
python -m unittest discover -s tests -v
```

CUDA resize/wavelet checks skip without a GPU. No checkpoint weights or private
benchmark footage are included. Upstream Apache-2.0 licence and assets are retained.
