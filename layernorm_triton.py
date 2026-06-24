import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel_optimized(
    input_ptr, gamma_ptr, beta_ptr, output_ptr,
    rows, cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < cols

    x = tl.load(input_ptr + row * cols + offs, mask=mask, other=0.0)

    # 两步规约，减少指令开销
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / cols
    x_centered = x - mean
    x_centered = tl.where(mask, x_centered, 0.0)

    var = tl.sum(x_centered * x_centered, axis=0) / cols
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)

    out = x_centered * inv_std * gamma + beta
    tl.store(output_ptr + row * cols + offs, out, mask=mask)


def triton_layernorm(input, gamma, beta, eps=1e-5):
    assert input.is_cuda
    assert input.dim() == 2
    assert input.dtype == torch.float32
    assert input.is_contiguous()

    rows, cols = input.shape
    output = torch.empty_like(input)

    BLOCK_SIZE = triton.next_power_of_2(cols)
    if BLOCK_SIZE < 128:
        BLOCK_SIZE = 128

    grid = (rows,)
    layernorm_kernel_optimized[grid](
        input, gamma, beta, output,
        rows, cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    return output