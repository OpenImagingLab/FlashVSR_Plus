"""Optional TC4 conditioning storage for the official FlashVSR+ pipeline.

The caller must budget this full-clip BF16 GPU allocation alongside model/KV
memory. It is not an automatic low-VRAM policy. Pass ``padded`` to StreamRunner.run(padded_lq=...) alongside ``video``.

PIL resizing is the default. The opt-in two-pass tensor resize approximates
PIL's intermediate uint8 rounding; it is not claimed to be pixel-identical.
Already decoded RGB8 frames can be passed directly, without an MP4 round trip.
Crop geometry follows OpenImagingLab/FlashVSR_Plus preprocessing (Apache-2.0).
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def padded_frame_capacity(frame_count):
    if frame_count < 1:
        raise ValueError('At least one source frame is required')
    effective = ((frame_count - 1 + 3) // 4) * 4 + 1
    latent_count = (effective - 1) // 4 + 1
    return 1 + ((latent_count + 3) // 4) * 16


def resident_bytes(frame_count, height, width):
    if height < 1 or width < 1:
        raise ValueError('Target dimensions must be positive')
    return 3 * padded_frame_capacity(frame_count) * height * width * 2


def _tensor_resize(batch, height, width):
    source_height, source_width = batch.shape[-2:]
    if (source_height, source_width) == (height, width):
        return batch
    if source_height / source_width < height / width:
        crop_width = int(source_height / height * width)
        left = (source_width - crop_width) // 2
        batch = batch[:, :, :, left:left + crop_width]
    else:
        crop_height = int(source_width / width * height)
        top = (source_height - crop_height) // 2
        batch = batch[:, :, top:top + crop_height, :]
    # PIL RGB resampling materializes/rounds the horizontal pass before the
    # vertical pass. One fused 2D interpolation has different rounding.
    for size in [(batch.shape[-2], width), (height, width)]:
        batch = F.interpolate(batch, size=size, mode='bicubic',
                              align_corners=False, antialias=True)
        batch.add_(.5).floor_().clamp_(0, 255)
    return batch


@dataclass
class ResidentInput:
    video: torch.Tensor
    padded: torch.Tensor
    reference_rgb8: np.ndarray
    resize_control: dict | None


@torch.inference_mode()
def prepare_rgb8(frames, height, width, *, device='cuda', batch_frames=4,
                 tensor_resize=False, max_resident_bytes=None):
    """Prepare [F,H,W,3] uint8 RGB, retaining colour reference on CPU.

    ``max_resident_bytes`` limits the input allocation, not total model/workspace
    VRAM. Unsupported or numerically divergent tensor resizing raises so the
    caller can retry explicitly with PIL. Model weights/attention are untouched.
    """
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8
            or frames.ndim != 4 or frames.shape[-1] != 3
            or not frames.shape[1] or not frames.shape[2]):
        raise ValueError('Expected nonempty [F,H,W,3] uint8 RGB frames')
    if height <= 0 or width <= 0:
        raise ValueError('Target height and width must be positive')
    if batch_frames < 1:
        raise ValueError('batch_frames must be positive')
    count = len(frames)
    allocation = resident_bytes(count, height, width)
    if max_resident_bytes is not None and allocation > max_resident_bytes:
        raise ValueError(f'Resident input requires {allocation} bytes, exceeding its budget')
    from inference.preprocessing.input import pil_center_crop_resize
    if not tensor_resize:
        frames = pil_center_crop_resize(frames, height, width,
                                        resample=Image.Resampling.BICUBIC)
    reference = (np.empty((count, height, width, 3), dtype=np.uint8)
                 if tensor_resize else frames)
    capacity = padded_frame_capacity(count)
    storage = torch.empty((1, 3, capacity, height, width),
                          dtype=torch.bfloat16, device=device)
    control = None
    for start in range(0, count, batch_frames):
        end = min(count, start + batch_frames)
        # Transfer bytes first. A blocking combined device+dtype transfer can
        # otherwise perform the float conversion on the host in PyTorch.
        batch = torch.from_numpy(np.ascontiguousarray(frames[start:end]))
        batch = batch.to(device=device, non_blocking=True).permute(0, 3, 1, 2).float()
        if tensor_resize:
            batch = _tensor_resize(batch, height, width)
            rgb8 = batch.permute(0, 2, 3, 1).byte().contiguous().cpu().numpy()
            reference[start:end] = rgb8
            if start == 0:
                expected = pil_center_crop_resize(frames[start:end], height, width,
                                                  resample=Image.Resampling.BICUBIC)
                difference = np.abs(rgb8.astype(np.int16) - expected.astype(np.int16))
                control = {'maxRgb8Difference': int(difference.max()),
                           'meanRgb8Difference': float(difference.mean()),
                           'frames': len(rgb8)}
                if control['maxRgb8Difference'] > 2 or control['meanRgb8Difference'] > .3:
                    raise ValueError(f'Tensor resize failed PIL numeric control: {control}')
        batch.div_(127.5).sub_(1)
        storage[:, :, start:end].copy_(batch.permute(1, 0, 2, 3).unsqueeze(0))
    # A cloned final frame avoids overlapping-source copy restrictions.
    storage[:, :, count:].copy_(storage[:, :, count - 1:count].clone().expand(
        -1, -1, capacity - count, -1, -1))
    return ResidentInput(storage[:, :, :count], storage, reference, control)
