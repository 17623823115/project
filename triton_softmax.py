import torch
import triton
import triton.language as tl


# ===================== Unmasked Softmax (compatible mode retained) =====================
@triton.jit
def _softmax_row_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = row_id * stride_row
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(input_ptr + row_start + offs, mask=mask, other=-float('inf'))
    
    max_val = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    inv_sum = 1.0 / sum_exp
    out = exp_x * inv_sum
    
    tl.store(output_ptr + row_start + offs, out, mask=mask)


# ===================== New addition: Mask fusion Softmax =====================
@triton.jit
def _softmax_row_masked_kernel(
    input_ptr, mask_ptr, output_ptr,
    M, N,
    stride_row,
    stride_mask_row,
    BLOCK_N: tl.constexpr,
):
    """Mask fusion version: input + mask then directly apply Softmax, completed in a single kernel"""
    row_id = tl.program_id(0)
    row_start = row_id * stride_row
    mask_start = row_id * stride_mask_row
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(input_ptr + row_start + offs, mask=mask, other=-float('inf'))
    mask_val = tl.load(mask_ptr + mask_start + offs, mask=mask, other=-float('inf'))
    x = x + mask_val  # Mask directly added in registers, no intermediate result written back
    
    max_val = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    inv_sum = 1.0 / sum_exp
    out = exp_x * inv_sum
    
    tl.store(output_ptr + row_start + offs, out, mask=mask)


class Softmax(torch.nn.Module):
    """
    Completely compatible with the torch.nn.Softmax(dim=-1) interface
    Supports an optional mask parameter. If passed, it will use mask-based fusion with a single kernel.
    """
    use_triton = False

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        assert dim == -1, "Only supports dim=-1"

    def forward(self, x, mask=None):
        if not Softmax.use_triton:
            # Original PyTorch path
            if mask is not None:
                x = x + mask
            return torch.nn.functional.softmax(x, dim=self.dim)
        else:
            assert x.is_cuda and x.dtype == torch.float32
            x = x.contiguous()
            N = x.shape[-1]
            M = x.numel() // N
            output = torch.empty_like(x)

            # Automatic block parameter adaptation
            if N <= 32:
                BLOCK_N, num_warps = 32, 1
            elif N <= 64:
                BLOCK_N, num_warps = 64, 2
            elif N <= 128:
                BLOCK_N, num_warps = 128, 4
            elif N <= 256:
                BLOCK_N, num_warps = 256, 4
            elif N <= 1024:
                BLOCK_N, num_warps = 1024, 8
            else:
                BLOCK_N, num_warps = 1024, 8

            grid = (M,)
            if mask is not None:
                # Mask fusion path
                mask = mask.contiguous()
                _softmax_row_masked_kernel[grid](
                    x, mask, output, M, N,
                    x.stride(-2), mask.stride(-2),
                    BLOCK_N=BLOCK_N,
                    num_warps=num_warps,
                )
            else:
                # Unmasked original path
                _softmax_row_kernel[grid](
                    x, output, M, N,
                    x.stride(-2),
                    BLOCK_N=BLOCK_N,
                    num_warps=num_warps,
                )
            return output