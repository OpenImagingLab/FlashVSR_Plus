<div align="center">

<h1>⚡ FlashVSR+</h1>

<p>
  <strong>Ruchang Yao<sup>1,2</sup> &nbsp; Shi Guo<sup>1,†</sup> &nbsp; Junhao Zhuang<sup>3</sup> &nbsp; Mingde Yao<sup>2,4</sup> &nbsp; Xin Cai<sup>1,2</sup></strong><br>
  <strong>Jiahao Zhan<sup>2</sup> &nbsp; Bin Fu<sup>1</sup> &nbsp; Xinyuan Chen<sup>1</sup> &nbsp; Yu Qiao<sup>1</sup> &nbsp; Tianfan Xue<sup>1,2,4,†</sup></strong>
</p>

<p>
  <sup>1</sup>Shanghai AI Lab &nbsp; <sup>2</sup>MMLAB CUHK &nbsp; <sup>3</sup>Tsinghua University &nbsp; <sup>4</sup>CPII under InnoHK<br>
  <sup>†</sup> Corresponding authors.
</p>

<p>
  <a href="https://flashvsr.tech"><img src="https://img.shields.io/badge/Project-333333?style=flat&amp;logo=googlechrome&amp;logoColor=white" alt="Project"></a>
  &nbsp;
  <a href="https://huggingface.co/ruchangyao/FlashVSR_plus"><img src="https://img.shields.io/badge/Models-333333?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Models"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/ruchangyao/FlashVSR_Plus_Dataset"><img src="https://img.shields.io/badge/Dataset-333333?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Dataset"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2610.17552"><img src="https://img.shields.io/badge/arXiv-333333?style=flat&amp;logo=arxiv&amp;logoColor=white" alt="arXiv"></a>
</p>

<p>
  <a href="#installation">Installation</a> ·
  <a href="#model-files">Model files</a> ·
  <a href="#command-line-inference">Run inference</a> ·
  <a href="#python-api">Python API</a>
</p>

</div>

**Your star means a lot for us to develop this project!** :star:
## Overview

[![FlashVSR+ teaser: visual comparisons and inference performance](assets/teaser.png)](assets/teaser.pdf)

**FlashVSR+** is an efficient one-step diffusion framework for high-quality video super-resolution, enabling **spatially untiled 4K inference on a single 24GB RTX 4090 GPU**.

FlashVSR+ introduces two key innovations:

- **Hybrid Attention:** Exploits spatial and temporal attention locality while retaining sparse global interactions, achieving up to **2× DiT acceleration** and **45% lower peak memory**.
- **Reward-Yield Distribution Matching Distillation (RY-DMD):** Enables preference alignment with black-box rewards without differentiating through the reward model, improving perceptual quality and generalizing to text-to-image generation.

We release the inference code, pretrained models, and training dataset to facilitate further research and applications.

## Highlights

- **Three presets:** run Turbo, Base, or V1 through the same CLI and Python API. V1 uses the same inference configuration as Base with its own checkpoints.
- **Large-motion support:** lower `--temporal-compression` from `4` (default) to `2` or `1` to better preserve motion details in videos with large motion.
- **Long-video option:** a numeric `--decode-every` reads and writes video in segments; the default `auto` mode loads the full input.
- **Attention backends:** use Block Sparse Attention, the included Triton operator, or PyTorch SDPA.

## Third-Party Integrations

We appreciate the community's efforts to extend and optimize FlashVSR+:

- **[fal.ai](https://fal.ai/models/openimaginglab/flashvsr-plus)** — Online inference API.
- **[FlashVSR+ ComfyUI](https://huggingface.co/t8star/Flashvsr-Plus-Comfy)** — Community ComfyUI integration.
- **[FlashVSRPlusTiledUpscaleVideo](https://comfy.icu/node/FlashVSRPlusTiledUpscaleVideo)** — ComfyUI video upscaling node.
- **[RTX 5090 Performance Optimization](https://github.com/OpenImagingLab/FlashVSR_Plus/pull/3)** — Community-contributed inference optimizations by [5andr0](https://github.com/5andr0), reducing CPU transfers and improving buffer management.

**Disclaimer:** These community contributions have not been officially tested or verified by our team. We sincerely thank all contributors for their valuable work!

## Requirements

- Linux
- NVIDIA GPU with CUDA
- Conda/Miniconda or Python `venv`
- Python 3.10
- Input height and width must both be divisible by 128

The released inference pipeline is CUDA-only and does not support CPU inference.

## Installation

### 1. Create the environment

#### Option A: Conda (recommended)

```bash
conda create -n flashvsr_plus python=3.10 -y
conda activate flashvsr_plus

pip install -r requirements.txt
pip install -e . --no-deps
```

The editable installation exposes the `flashvsr_plus` Python package and keeps
the `python -m inference.run_stream` command available.

`--no-deps` is intentional. It prevents the package installation from modifying an existing PyTorch/CUDA environment.

#### Option B: Python `venv`

The project also works in a standard Python 3.10 virtual environment. It uses
the host NVIDIA driver/CUDA runtime.

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

After either setup, activate the environment before installing an optional
attention backend or running inference.

### 2. Install an attention backend

FlashVSR+ supports two optimized sparse-attention backends:

1. **Block Sparse Attention (BSA)** from MIT Han Lab — recommended for NVIDIA data-center GPUs such as **A100, H100, and H200**, where the optimized CUDA operator is supported.
2. **FlashVSR Triton Attention** — our custom Triton implementation, compatible with a wide range of NVIDIA GPUs, including **RTX 4090, A100, H100, and H200**. It is particularly recommended for **RTX 4090** and other consumer GPUs where BSA may be unavailable.

PyTorch SDPA is also available as a fallback.

The automatic backend priority is:

```text
BSA → FlashVSR Triton → PyTorch SDPA
```

Only one optimized backend needs to be installed.

#### Option A: Block Sparse Attention (recommended)

Clone the official Block-Sparse-Attention repository:

```bash
cd operators
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
```

Then follow the installation instructions in the official BSA repository to build and install the CUDA extension.

BSA requires a working CUDA compilation toolchain, including `nvcc`.

After installation, FlashVSR+ will automatically use BSA when it is available and compatible with the current GPU.

#### Option B: FlashVSR Triton Attention

Our Triton backend depends on FlashAttention utilities. Install FlashAttention first:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Then install the FlashVSR Triton operator:

```bash
pip install -e operators/flashvsr_attn --no-build-isolation --no-deps
```

Unlike BSA, the Triton backend does not compile a custom CUDA extension with `nvcc`.

If neither optimized backend is available, FlashVSR+ falls back to PyTorch SDPA.

To explicitly force SDPA:

```bash
python -m inference.run_stream ... --hardware-profile generic
```

You can also explicitly select an attention backend using `--attn-kernel`. If the requested backend is unavailable, inference will fail early instead of silently switching implementations.

## Model Files

Download the Turbo and Base model weights from
[ruchangyao/FlashVSR_plus on Hugging Face](https://huggingface.co/ruchangyao/FlashVSR_plus/tree/main).

For the `v1` preset, download the weights separately from
[JunhaoZhuang/FlashVSR-v1.1 on Hugging Face](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1).
Place `dit.safetensors` and `lq_proj_in.safetensors` in `ckpts/v1/`.

After downloading, arrange the files under the repository root as shown below.
Keep the Turbo, Base, and V1 checkpoints in their respective directories; all
presets share `TCDecoder.ckpt`.

```text
ckpts/
├── turbo/
│   ├── dit.safetensors
│   └── lq_proj_in.safetensors
├── base/
│   ├── dit.safetensors
│   └── lq_proj_in.safetensors
├── v1/
│   ├── dit.safetensors
│   └── lq_proj_in.safetensors
├── TCDecoder.ckpt
└── Wan2.1_VAE.pth          # only required when using the VAE decoder

prompt_tensor/
└── posi_prompt.pth
```

The default streaming pipeline uses `TCDecoder.ckpt`.

`prompt_tensor/posi_prompt.pth` is included in this code repository. Keep it at
the path shown above when using the default presets.

`Wan2.1_VAE.pth` is only required when using the VAE decoder through the unified Python API or `inference.cli`.

## Command-Line Inference

Run from the repository root after installing dependencies and placing the
checkpoints under `ckpts/`.

### Turbo

```bash
python -m inference.run_stream \
    --input examples/case_0.mp4 \
    --mode turbo \
    --output outputs/case_0_turbo.mp4 \
    --height 2048 \
    --width 3840
```

### Base

```bash
python -m inference.run_stream \
    --input examples/case_1.mp4 \
    --mode base \
    --output outputs/case_1_base.mp4 \
    --height 1024 \
    --width 1920
```

### V1

V1 uses the same inference configuration as Base and loads weights from
`ckpts/v1/`.

```bash
python -m inference.run_stream \
    --input examples/case_1.mp4 \
    --mode v1 \
    --output outputs/case_1_v1.mp4 \
    --height 1024 \
    --width 1920
```

### Incremental input and decoding

```bash
python -m inference.run_stream \
    --input examples/case_2.mp4 \
    --mode turbo \
    --output outputs/case_2_turbo.mp4 \
    --height 2176 \
    --width 3840 \
    --decode-every 28
```

`--decode-every` takes a positive multiple of 4. Smaller values reduce the number of frames held for each decoding segment. Without this option, the default `auto` mode loads the full input video before inference.

Both `height` and `width` must be divisible by 128.

The CLI center-crops and resizes the input, then writes an H.264 MP4. It keeps the input FPS and audio stream when present. Incomplete temporal groups are padded internally and cropped back to the original frame count.

### Temporal compression

`--temporal-compression` controls the effective number of input frames per VAE latent frame: `4` (default), `2`, or `1`. For videos with large motion, use `2` or `1` to better preserve motion details, at the cost of increased inference time. We recommend `4` for most videos, and `2` or `1` for large-motion videos.

## CLI Options

| Option | Default | Description |
|---|---:|---|
| `--mode {turbo,base,v1}` | `turbo` | Select the model preset. |
| `--input PATH` | required | Input video path. |
| `--output PATH` | required | Output MP4 path. |
| `--height N` | required | Output height. Must be divisible by 128. |
| `--width N` | required | Output width. Must be divisible by 128. |
| `--max-frames N` | all | Maximum number of decoded input frames. |
| `--cache-offload-mode {auto,never,always}` | `auto` | KV-cache residency policy. |
| `--decode-every N` | `auto` | Read and decode in segments. N must be a positive multiple of 4; `auto` loads the full input. |
| `--temporal-compression {1,2,4}` | `4` | Input frames per latent frame. Use `2` or `1` for large motion, with higher inference cost. |
| `--cf` | off | Enable low-frequency color correction. |

For all available options, run `python -m inference.run_stream --help`.

<details>
<summary>Advanced options</summary>

| Option | Default | Description |
|---|---:|---|
| `--seed N` | `1` | Latent initialization seed. |
| `--device DEVICE` | `cuda:0` | CUDA device used for inference. |
| `--ckpt-dir PATH` | preset path | Override the checkpoint directory for the selected model. |
| `--hardware-profile {auto,generic}` | `auto` | `auto` enables optimized kernels; `generic` forces PyTorch SDPA. |
| `--attn-kernel NAME` | `auto` | Stream attention backend: `auto`, `sdpa`, `block_sparse`, or `triton_block_sparse`. FlashAttention 2/3 cannot represent the required causal/sparse block masks. |
| `--tiny-decoder-mode {auto,serial,parallel}` | `auto` | TinyDecoder execution strategy. Serial mode uses less workspace memory. |
| `--serial-tiny-decoder` | off | Shorthand for serial TinyDecoder execution. |
| `--tiny-decoder-channels-last/--no-tiny-decoder-channels-last` | on | Use channels-last BF16 convolution storage. |
| `--color-correction/--no-color-correction` | off | Enable or disable low-frequency color correction; `--cf` enables it. |
| `--color-correction-levels N` | `5` | Wavelet color-correction levels. |
| `--color-correction-chunk-size N` | `16` | Frames processed per color-correction chunk. |
| `--offload-original-cache` | off | Offload selected original-attention KV caches to host memory. |
| `--temporal-merge {gaussian,first}` | `gaussian` | Strategy for merging repeated frames at temporal compression 1 or 2. |
| `--temporal-gaussian-sigma N` | `1.0` | Gaussian sigma used for temporal merging. |
| `--video-preset NAME` | `veryfast` | x264 encoding preset, from `ultrafast` to `slow`. |

</details>

## Attention Backend Examples

Force BSA:

```bash
python -m inference.run_stream ... \
    --attn-kernel block_sparse
```

Force the FlashVSR Triton backend:

```bash
python -m inference.run_stream ... \
    --attn-kernel triton_block_sparse
```

Force the portable PyTorch SDPA implementation:

```bash
python -m inference.run_stream ... \
    --hardware-profile generic
```

## Python API

Run this example from the repository root with the model files in place.

```python
from pathlib import Path

from inference import InferenceRequest, build_inference_pipeline
from inference.presets import model_paths_for_mode, stream_config_for_mode

config = stream_config_for_mode("turbo", device="cuda:0")
paths = model_paths_for_mode(Path.cwd(), "turbo")
pipeline = build_inference_pipeline(config, paths)

result = pipeline.infer(
    InferenceRequest(
        input_path="input.mp4",
        output_path="outputs/restored.mp4",
        height=384,
        width=768,
    )
)
```

The same public API is also exposed through the stable package namespace:

```python
from flashvsr_plus import InferenceRequest, build_inference_pipeline
```

Tensor inputs must have one of the following shapes:

```text
[1, 3, F, H, W]
[1, F, 3, H, W]
```

Input values should use the model's `[-1, 1]` range.

The context tensor can either be supplied for each request or loaded from the model bundle.

## Repository Layout

```text
inference/       CLI, inference pipeline, preprocessing, and decoding
models/          DiT, attention, conditioning, and VAE implementations
model_io/        checkpoint loading and key normalization
operators/       optional attention operators
prompt_tensor/   released context tensor
ckpts/           local model weights
```

## Citation

```bibtex
@misc{yao2026flashvsrefficientvideosuperresolution,
      title={FlashVSR+: Efficient Video Super-Resolution via Hybrid Attention and Reward-Yield Distribution Matching}, 
      author={Ruchang Yao and Shi Guo and Junhao Zhuang and Mingde Yao and Xin Cai and Jiahao Zhan and Bin Fu and Xinyuan Chen and Yu Qiao and Tianfan Xue},
      year={2026},
      eprint={2610.17552},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.17552}, 
}

@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}
```

## Star History

<a href="https://www.star-history.com/?repos=OpenImagingLab%2FFlashVSR_Plus&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=OpenImagingLab/FlashVSR_Plus&type=date&theme=dark&legend=top-left" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=OpenImagingLab/FlashVSR_Plus&type=date&legend=top-left" />
    <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=OpenImagingLab/FlashVSR_Plus&type=date&legend=top-left" />
  </picture>
</a>
