"""FP32 replicate-border 3x3 wavelet blur without depthwise-convolution workspaces."""
import torch
import triton
import triton.language as tl

@triton.jit
def _blur(X,Y,H:tl.constexpr,W:tl.constexpr,N:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr):
 index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 plane=index//(H*W);y=index//W%H;x=index%W
 ym=tl.maximum(y-R,0);yp=tl.minimum(y+R,H-1);xm=tl.maximum(x-R,0);xp=tl.minimum(x+R,W-1)
 base=plane*H*W;valid=index<N
 a=tl.load(X+base+ym*W+xm,valid,0).to(tl.float32)*0.0625
 a=a+tl.load(X+base+ym*W+x,valid,0).to(tl.float32)*0.125
 a=a+tl.load(X+base+ym*W+xp,valid,0).to(tl.float32)*0.0625
 a=a+tl.load(X+base+y*W+xm,valid,0).to(tl.float32)*0.125
 a=a+tl.load(X+base+y*W+x,valid,0).to(tl.float32)*0.25
 a=a+tl.load(X+base+y*W+xp,valid,0).to(tl.float32)*0.125
 a=a+tl.load(X+base+yp*W+xm,valid,0).to(tl.float32)*0.0625
 a=a+tl.load(X+base+yp*W+x,valid,0).to(tl.float32)*0.125
 a=a+tl.load(X+base+yp*W+xp,valid,0).to(tl.float32)*0.0625
 tl.store(Y+index,a,valid)

def wavelet_blur(value,radius):
 if value.device.type!='cuda' or value.dtype!=torch.float32:raise ValueError('FP32 CUDA input required')
 source=value.contiguous();output=torch.empty_like(source)
 _blur[(triton.cdiv(source.numel(),1024),)](source,output,source.shape[-2],source.shape[-1],source.numel(),radius,1024,enable_fp_fusion=False)
 return output


_validated_devices = set()


def validate_wavelet(device):
    """Check first-use FP32 arithmetic against the stock blur before opting in."""
    key = str(torch.device(device))
    if key in _validated_devices:
        return
    from .color_correction import _wavelet_blur
    # Use a private generator so validation cannot perturb inference randomness.
    generator = torch.Generator(device=device).manual_seed(42)
    value = torch.randn((2, 3, 257, 259), generator=generator,
                        device=device, dtype=torch.float32)
    with torch.backends.cudnn.flags(allow_tf32=False):
        error = max(float((_wavelet_blur(value, r) - wavelet_blur(value, r)).abs().max())
                    for r in (1, 2, 4, 8, 16))
    if error > 2e-6:
        raise RuntimeError(f"Triton wavelet numeric control failed: max difference {error}")
    _validated_devices.add(key)
