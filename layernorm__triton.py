import torch
import triton
import triton.language as tl


#  General version 
@triton.jit
def layernorm_kernel_generic(
    input_ptr, gamma_ptr, beta_ptr, output_ptr,
    rows, cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < cols
    x = tl.load(input_ptr + row * cols + offs, mask=mask, other=0.0)
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


# 128-dim specialized version
@triton.jit
def layernorm_kernel_128(
    input_ptr, gamma_ptr, beta_ptr, output_ptr,
    rows, eps,
    BLOCK_SIZE: tl.constexpr = 128
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    base = row * BLOCK_SIZE

    x = tl.load(input_ptr + base + offs)
    sum_x = tl.sum(x, axis=0)
    mean = sum_x * 0.0078125  # 1/128
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) * 0.0078125
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs)
    beta = tl.load(beta_ptr + offs)
    out = x_centered * inv_std * gamma + beta
    tl.store(output_ptr + base + offs, out)


# 384-dim specialized version
@triton.jit
def layernorm_kernel_384(
    input_ptr, gamma_ptr, beta_ptr, output_ptr,
    rows, eps,
    BLOCK_SIZE: tl.constexpr = 512,
    COLS: tl.constexpr = 384
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < COLS
    base = row * COLS

    x = tl.load(input_ptr + base + offs, mask=mask, other=0.0)
    sum_x = tl.sum(x, axis=0)
    inv_cols = 1.0 / 384.0
    mean = sum_x * inv_cols
    x_centered = x - mean
    x_centered = tl.where(mask, x_centered, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) * inv_cols
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    out = x_centered * inv_std * gamma + beta
    tl.store(output_ptr + base + offs, out, mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    """
    LayerNorm Triton implementation
    Automatically identify 128/384 dimensions and call specialized kernels, other dimensions use the generic version
    """
    assert x.is_cuda
    assert x.dtype == torch.float32
    
    orig_shape = x.shape
    C = orig_shape[-1]
    M = x.numel() // C
    x_flat = x.reshape(M, C).contiguous()
    output_flat = torch.empty_like(x_flat)
    grid = (M,)

    # Fixed dimension with specialized core, achieving optimal performance
    if C == 128:
        layernorm_kernel_128[grid](
            x_flat, weight, bias, output_flat,
            M, eps,
            num_warps=4
        )
    elif C == 384:
        layernorm_kernel_384[grid](
            x_flat, weight, bias, output_flat,
            M, eps,
            num_warps=8
        )
    else:
        # Other dimensions use the generic version as a fallback
        BLOCK_SIZE = triton.next_power_of_2(C)
        if BLOCK_SIZE < 128:
            BLOCK_SIZE = 128
        layernorm_kernel_generic[grid](
            x_flat, weight, bias, output_flat,
            M, C, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
    
    return output_flat.reshape(orig_shape)