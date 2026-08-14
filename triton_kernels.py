import torch
import triton
import triton.language as tl


# ===================== Original GEMM (retaining compatibility) =====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel_with_bias(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_RELU: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_am[:, None] < M) & (k + offs_k[None, :] < K)
        mask_b = (k + offs_k[:, None] < K) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    bias = tl.load(Bias_ptr + offs_bn, mask=offs_bn < N)
    acc += bias[None, :]
    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear_gemm(x, B, bias, relu=False, allow_tf32=True):
    """New version: y = x @ B + bias (+ optional ReLU)"""
    assert x.is_cuda and B.is_cuda and bias.is_cuda
    assert x.shape[1] == B.shape[0]
    assert x.dtype == torch.float32 and B.dtype == torch.float32
    M, K = x.shape
    _, N = B.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    gemm_kernel_with_bias[grid](
        x, B, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        B.stride(0), B.stride(1),
        y.stride(0), y.stride(1),
        HAS_RELU=relu,
        ALLOW_TF32=allow_tf32,
    )
    return y


# ===================== New addition: Residual Fusion GEMM (quad-core kernel) =====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel_bias_residual(
    A_ptr, B_ptr, Bias_ptr, Residual_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_rm, stride_rn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_RELU: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """GEMM + Bias + ReLU +Residual addition
    Calculation: output = (x @ W.T + bias).relu() + residual
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N)
    acc += bias[None, :]

    # choosable ReLU
    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    # Add residual (completed in registers, not written back to memory)
    mask_r = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    residual = tl.load(Residual_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn, mask=mask_r, other=0.0)
    acc += residual

    # Write back the final result
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear_gemm_residual(x, B, bias, residual, relu=False, allow_tf32=True):
    """Residual fusion version: output = (x @ B + bias).relu() + residual"""
    assert x.is_cuda and B.is_cuda and bias.is_cuda and residual.is_cuda
    assert x.shape[1] == B.shape[0]
    assert x.shape[0] == residual.shape[0] and B.shape[1] == residual.shape[1]
    assert x.dtype == torch.float32 and B.dtype == torch.float32 and residual.dtype == torch.float32

    M, K = x.shape
    _, N = B.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    gemm_kernel_bias_residual[grid](
        x, B, bias, residual, y,
        M, N, K,
        x.stride(0), x.stride(1),
        B.stride(0), B.stride(1),
        residual.stride(0), residual.stride(1),
        y.stride(0), y.stride(1),
        HAS_RELU=relu,
        ALLOW_TF32=allow_tf32,
    )
    return y


# ==================== Fusion of attention: QK dot product + bias + mask + Softmax ====================
@triton.jit
def fused_attention_softmax_kernel(
    q_ptr, k_ptr, extra_att_ptr, mask_ptr, out_ptr,
    N, C,
    scalar_scale,
    stride_q_m, stride_q_n, stride_q_c,
    stride_k_m, stride_k_n, stride_k_c,
    stride_att_m, stride_att_n, stride_att_j,
    stride_mask_m, stride_mask_n, stride_mask_j,
    stride_out_m, stride_out_n, stride_out_j,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    head_idx = row_idx // N
    seq_idx = row_idx % N
    n_offs = tl.arange(0, BLOCK_N)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    q_base = q_ptr + head_idx * stride_q_m + seq_idx * stride_q_n
    q_val = tl.load(q_base + c_offs * stride_q_c, mask=c_mask, other=0.0)
    k_head_base = k_ptr + head_idx * stride_k_m

    # First round: Find the maximum value
    max_val = -float('inf')
    for n_start in range(0, N, BLOCK_N):
        n_cur = n_start + n_offs
        n_mask = n_cur < N
        k_block = tl.load(
            k_head_base + n_cur[:, None] * stride_k_n + c_offs[None, :] * stride_k_c,
            mask=n_mask[:, None] & c_mask[None, :],
            other=0.0
        )
        dot_val = tl.sum(q_val[None, :] * k_block, axis=1) * scalar_scale
        att_base = extra_att_ptr + head_idx * stride_att_m + seq_idx * stride_att_n
        extra_val = tl.load(att_base + n_cur * stride_att_j, mask=n_mask, other=0.0)
        dot_val += extra_val
        mask_base = mask_ptr + head_idx * stride_mask_m + seq_idx * stride_mask_n
        mask_val = tl.load(mask_base + n_cur * stride_mask_j, mask=n_mask, other=-float('inf'))
        dot_val += mask_val
        max_val = tl.maximum(max_val, tl.max(tl.where(n_mask, dot_val, -float('inf'))))

    # Second round: Calculate the exponents and
    sum_exp = 0.0
    out_base = out_ptr + head_idx * stride_out_m + seq_idx * stride_out_n
    for n_start in range(0, N, BLOCK_N):
        n_cur = n_start + n_offs
        n_mask = n_cur < N
        k_block = tl.load(
            k_head_base + n_cur[:, None] * stride_k_n + c_offs[None, :] * stride_k_c,
            mask=n_mask[:, None] & c_mask[None, :],
            other=0.0
        )
        dot_val = tl.sum(q_val[None, :] * k_block, axis=1) * scalar_scale
        att_base = extra_att_ptr + head_idx * stride_att_m + seq_idx * stride_att_n
        extra_val = tl.load(att_base + n_cur * stride_att_j, mask=n_mask, other=0.0)
        dot_val += extra_val
        mask_base = mask_ptr + head_idx * stride_mask_m + seq_idx * stride_mask_n
        mask_val = tl.load(mask_base + n_cur * stride_mask_j, mask=n_mask, other=-float('inf'))
        dot_val += mask_val
        exp_val = tl.exp(dot_val - max_val)
        sum_exp += tl.sum(tl.where(n_mask, exp_val, 0.0))
        tl.store(out_base + n_cur * stride_out_j, exp_val, mask=n_mask)

    # Third time: Normalization
    inv_sum = 1.0 / sum_exp
    for n_start in range(0, N, BLOCK_N):
        n_cur = n_start + n_offs
        n_mask = n_cur < N
        exp_val = tl.load(out_base + n_cur * stride_out_j, mask=n_mask, other=0.0)
        tl.store(out_base + n_cur * stride_out_j, exp_val * inv_sum, mask=n_mask)


def triton_fused_attention_softmax(q, k, extra_att, mask, scalar_scale):
    """Fusion version of attention calculation: QK dot product + bias叠加 + mask + Softmax single kernel completion"""
    assert q.is_cuda and k.is_cuda and extra_att.is_cuda and mask.is_cuda
    assert q.shape == k.shape
    assert extra_att.shape == mask.shape
    M, N, C = q.shape
    out = torch.empty((M, N, N), device=q.device, dtype=q.dtype)
    BLOCK_N = triton.next_power_of_2(min(N, 128))
    BLOCK_C = triton.next_power_of_2(C)
    grid = (M * N,)

    fused_attention_softmax_kernel[grid](
        q, k, extra_att, mask, out,
        N, C,
        scalar_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        extra_att.stride(0), extra_att.stride(1), extra_att.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
        num_warps=4
    )
    return out

    # ===================== Four-in-one integrated core: Residual + LayerNorm + Linear + ReLU =====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N_OUT', 'K'],
)
@triton.jit
def fused_residual_ln_linear_relu_kernel(
    s_ptr, ipa_ptr,
    ln_gamma_ptr, ln_beta_ptr,
    lin_weight_ptr, lin_bias_ptr,
    ln_out_ptr,    # New addition: LayerNorm output (for subsequent residual connection)
    lin_out_ptr,   # Linear+ReLU output
    M, K, N_OUT, eps,
    stride_s_m, stride_s_k,
    stride_ipa_m, stride_ipa_k,
    stride_lw_k, stride_lw_n,
    stride_ln_m, stride_ln_k,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fusion Computing:
        ln_out = LayerNorm(s + ipa_out)
        lin_out = ReLU( ln_out @ lin_weight + lin_bias )
    Intermediate value holding register. Only the final two results are written back to the memory.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # ---------- Step 1: Pre-calculate the mean and variance for each row ----------
    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask = (offs_m[:, None] < M) & mask_k[None, :]
        
        s_val = tl.load(s_ptr + offs_m[:, None] * stride_s_m + offs_k[None, :] * stride_s_k, mask=mask, other=0.0)
        ipa_val = tl.load(ipa_ptr + offs_m[:, None] * stride_ipa_m + offs_k[None, :] * stride_ipa_k, mask=mask, other=0.0)
        x_val = s_val + ipa_val
        
        sum_x += tl.sum(x_val, axis=1)
        sum_x2 += tl.sum(x_val * x_val, axis=1)
    
    mean = sum_x / K
    var = sum_x2 / K - mean * mean
    inv_std = tl.rsqrt(var + eps)
    
    # Step 2: Calculate LayerNorm for each block and write the results, 
    # while also completing the GEMM accumulation. 
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        
        # Load and calculate the residuals and
        s_val = tl.load(s_ptr + offs_m[:, None] * stride_s_m + offs_k[None, :] * stride_s_k, mask=mask_a, other=0.0)
        ipa_val = tl.load(ipa_ptr + offs_m[:, None] * stride_ipa_m + offs_k[None, :] * stride_ipa_k, mask=mask_a, other=0.0)
        x_val = s_val + ipa_val
        
        # Real-time computation of LayerNorm
        x_centered = x_val - mean[:, None]
        x_norm = x_centered * inv_std[:, None]
        gamma_val = tl.load(ln_gamma_ptr + offs_k, mask=mask_k, other=0.0)
        beta_val = tl.load(ln_beta_ptr + offs_k, mask=mask_k, other=0.0)
        a_val = x_norm * gamma_val[None, :] + beta_val[None, :]
        
        # Write the LayerNorm output (write only once, for subsequent residual use)
        ln_ptrs = ln_out_ptr + offs_m[:, None] * stride_ln_m + offs_k[None, :] * stride_ln_k
        tl.store(ln_ptrs, a_val, mask=mask_a)
        
        # Load weight blocks and accumulate GEMM
        mask_b = mask_k[:, None] & (offs_n[None, :] < N_OUT)
        b_val = tl.load(
            lin_weight_ptr + offs_k[:, None] * stride_lw_k + offs_n[None, :] * stride_lw_n,
            mask=mask_b, other=0.0
        )
        acc += tl.dot(a_val, b_val, allow_tf32=ALLOW_TF32)
    
    # ---------- Step 3: Bias + ReLU + Write back Linear results ----------
    bias_val = tl.load(lin_bias_ptr + offs_n, mask=offs_n < N_OUT)
    acc += bias_val[None, :]
    acc = tl.maximum(acc, 0.0)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    out_ptrs = lin_out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=mask_out)


def triton_fused_residual_ln_linear_relu(s, ipa_out, ln_weight, ln_bias, lin_weight, lin_bias, eps=1e-5, allow_tf32=True):
    """
    Four-in-one integrated interface
    Returns:
        ln_out: [M, K] LayerNorm output (used as subsequent residual source)
        lin_out: [M, N_OUT] Linear+ReLU output
    """
    assert s.is_cuda and ipa_out.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda
    assert lin_weight.is_cuda and lin_bias.is_cuda
    assert s.shape == ipa_out.shape
    assert s.shape[1] == ln_weight.shape[0] == lin_weight.shape[0]
    assert lin_weight.shape[1] == lin_bias.shape[0]
    assert s.dtype == torch.float32
    
    M, K = s.shape
    _, N_OUT = lin_weight.shape
    ln_out = torch.empty_like(s)
    lin_out = torch.empty((M, N_OUT), device=s.device, dtype=s.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N_OUT, meta['BLOCK_N']),
    )
    
    fused_residual_ln_linear_relu_kernel[grid](
        s, ipa_out,
        ln_weight, ln_bias,
        lin_weight, lin_bias,
        ln_out, lin_out,
        M, K, N_OUT, eps,
        s.stride(0), s.stride(1),
        ipa_out.stride(0), ipa_out.stride(1),
        lin_weight.stride(0), lin_weight.stride(1),
        ln_out.stride(0), ln_out.stride(1),
        lin_out.stride(0), lin_out.stride(1),
        ALLOW_TF32=allow_tf32,
    )
    return ln_out, lin_out

# ===================== Two-layer fused kernel: Linear2 + ReLU + Linear3 + Residual =====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K1': 32, 'BLOCK_K2': 128, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K1': 32, 'BLOCK_K2': 128, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K1': 32, 'BLOCK_K2': 128, 'GROUP_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K1': 32, 'BLOCK_K2': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'K1', 'K2', 'N'],
)
@triton.jit
def fused_two_linear_residual_kernel(
    x_ptr,
    w2_ptr, b2_ptr,
    w3_ptr, b3_ptr,
    residual_ptr,
    out_ptr,
    M, K1, K2, N,
    stride_x_m, stride_x_k1,
    stride_w2_k1, stride_w2_k2,
    stride_w3_k2, stride_w3_n,
    stride_r_m, stride_r_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fused computation: out = ( (x @ w2 + b2).relu() ) @ w3 + b3 + residual
    Intermediate activation results are all held in registers, with no additional global memory reads/writes
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k2 = tl.arange(0, BLOCK_K2)
    mask_k2 = offs_k2 < K2

    # ---------- Step 1: Calculate the first layer Linear + Bias, with intermediate results held in registers ----------
    hidden = tl.zeros((BLOCK_M, BLOCK_K2), dtype=tl.float32)
    for k1 in range(0, K1, BLOCK_K1):
        offs_k1 = k1 + tl.arange(0, BLOCK_K1)
        mask_k1 = offs_k1 < K1

        mask_x = (offs_m[:, None] < M) & mask_k1[None, :]
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_x_m + offs_k1[None, :] * stride_x_k1,
            mask=mask_x, other=0.0
        )

        mask_w2 = mask_k1[:, None] & mask_k2[None, :]
        w2_block = tl.load(
            w2_ptr + offs_k1[:, None] * stride_w2_k1 + offs_k2[None, :] * stride_w2_k2,
            mask=mask_w2, other=0.0
        )

        hidden += tl.dot(x_block, w2_block, allow_tf32=ALLOW_TF32)

    # Add bias + ReLU activation (performed within the register and not written back to the memory)
    b2_val = tl.load(b2_ptr + offs_k2, mask=mask_k2, other=0.0)
    hidden += b2_val[None, :]
    hidden = tl.maximum(hidden, 0.0)

    # ---------- Step 2: Calculate the second layer Linear + Bias + Residual ----------
    mask_w3 = mask_k2[:, None] & (offs_n[None, :] < N)
    w3_block = tl.load(
        w3_ptr + offs_k2[:, None] * stride_w3_k2 + offs_n[None, :] * stride_w3_n,
        mask=mask_w3, other=0.0
    )

    acc = tl.dot(hidden, w3_block, allow_tf32=ALLOW_TF32)

    # Add the second layer bias
    b3_val = tl.load(b3_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b3_val[None, :]

    # Add residual within registers
    mask_r = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    residual_val = tl.load(
        residual_ptr + offs_m[:, None] * stride_r_m + offs_n[None, :] * stride_r_n,
        mask=mask_r, other=0.0
    )
    acc += residual_val

    # Write back the final result once
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n,
        acc, mask=mask_r
    )


def triton_fused_two_linear_residual(x, w2, b2, w3, b3, residual, allow_tf32=True):
    """
    Two-layer fused kernel: Linear2 + ReLU + Linear3 + Residual
    Parameters:
        x: [M, K1] Input features
        w2: [K1, K2] Transposed weights for the first layer
        b2: [K2] Bias for the first layer
        w3: [K2, N] Transposed weights for the second layer
        b3: [N] Bias for the second layer
        residual: [M, N] Residual features
    Returns:
        out: [M, N] Fused computation result
    """
    assert x.is_cuda and w2.is_cuda and b2.is_cuda and w3.is_cuda and b3.is_cuda and residual.is_cuda
    assert x.shape[1] == w2.shape[0]
    assert w2.shape[1] == w3.shape[0] == b2.shape[0]
    assert w3.shape[1] == b3.shape[0] == residual.shape[1]
    assert x.shape[0] == residual.shape[0]
    assert x.dtype == torch.float32

    M, K1 = x.shape
    K2 = w2.shape[1]
    N = w3.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_two_linear_residual_kernel[grid](
        x,
        w2, b2,
        w3, b3,
        residual,
        out,
        M, K1, K2, N,
        x.stride(0), x.stride(1),
        w2.stride(0), w2.stride(1),
        w3.stride(0), w3.stride(1),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1),
        ALLOW_TF32=allow_tf32,
    )
    return out

# ===================== Integration core: Linear GEMM + Point Rigid Transformation (Rotation + Translation) =====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32, 'BLOCK_P': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_K': 64, 'BLOCK_P': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_K': 32, 'BLOCK_P': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_K': 64, 'BLOCK_P': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_K': 128, 'BLOCK_P': 16}, num_stages=4, num_warps=4),
    ],
    key=['K', 'HP'],
)
@triton.jit
def fused_linear_point_transform_kernel(
    x_ptr, weight_ptr, bias_ptr, rot_ptr, trans_ptr, out_ptr,
    M, K, HP,
    stride_x_m, stride_x_k,
    stride_w_k, stride_w_n,
    stride_rot_m,
    stride_trans_m,
    stride_out_m, stride_out_p,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fused computation: out = (x @ weight + bias).reshape(HP, 3) @ rot.T + trans
    Omit global memory reads/writes of intermediate local point coordinates, with all intermediate results held in registers
    """
    m = tl.program_id(0)         
    p_block_idx = tl.program_id(1) 
    p_start = p_block_idx * BLOCK_P
    p_offs = p_start + tl.arange(0, BLOCK_P)
    p_mask = p_offs < HP

    
    n0 = p_offs * 3
    n1 = p_offs * 3 + 1
    n2 = p_offs * 3 + 2

    
    acc0 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_P,), dtype=tl.float32)

    # ---------- Step 1: Linear GEMM computation (local coordinates, not written back to memory) ----------
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load the input feature block (one-dimensional vector)
        x_val = tl.load(x_ptr + m * stride_x_m + k_offs * stride_x_k, mask=k_mask, other=0.0)
        # Rearrange to a two-dimensional form [1, BLOCK_K], meeting the rank requirement of tl.dot
        x_val_2d = x_val[None, :]

        # Load three columns of weights, shape [BLOCK_K, BLOCK_P]
        w0 = tl.load(
            weight_ptr + k_offs[:, None] * stride_w_k + n0[None, :] * stride_w_n,
            mask=k_mask[:, None] & p_mask[None, :], other=0.0
        )
        w1 = tl.load(
            weight_ptr + k_offs[:, None] * stride_w_k + n1[None, :] * stride_w_n,
            mask=k_mask[:, None] & p_mask[None, :], other=0.0
        )
        w2 = tl.load(
            weight_ptr + k_offs[:, None] * stride_w_k + n2[None, :] * stride_w_n,
            mask=k_mask[:, None] & p_mask[None, :], other=0.0
        )

        # Matrix multiplication results in [1, BLOCK_P], and then reshaping it into a one-dimensional form and performing cumulative addition.
        acc0 += tl.reshape(tl.dot(x_val_2d, w0, allow_tf32=ALLOW_TF32), (BLOCK_P,))
        acc1 += tl.reshape(tl.dot(x_val_2d, w1, allow_tf32=ALLOW_TF32), (BLOCK_P,))
        acc2 += tl.reshape(tl.dot(x_val_2d, w2, allow_tf32=ALLOW_TF32), (BLOCK_P,))

    # Add Linear bias
    bias0 = tl.load(bias_ptr + n0, mask=p_mask, other=0.0)
    bias1 = tl.load(bias_ptr + n1, mask=p_mask, other=0.0)
    bias2 = tl.load(bias_ptr + n2, mask=p_mask, other=0.0)
    acc0 += bias0
    acc1 += bias1
    acc2 += bias2

    # ---------- Step 2: Rigid transformation (rotation + translation), completed in registers ----------
    # Load the 3x3 rotation matrix for the current residue (row-major storage)
    rot_base = rot_ptr + m * stride_rot_m
    r00 = tl.load(rot_base + 0)
    r01 = tl.load(rot_base + 1)
    r02 = tl.load(rot_base + 2)
    r10 = tl.load(rot_base + 3)
    r11 = tl.load(rot_base + 4)
    r12 = tl.load(rot_base + 5)
    r20 = tl.load(rot_base + 6)
    r21 = tl.load(rot_base + 7)
    r22 = tl.load(rot_base + 8)

    # Load the translation vector of the current residue
    trans_base = trans_ptr + m * stride_trans_m
    t0 = tl.load(trans_base + 0)
    t1 = tl.load(trans_base + 1)
    t2 = tl.load(trans_base + 2)

    # Local coordinates @ Transposed rotation matrix + Translation vector
    out0 = acc0 * r00 + acc1 * r10 + acc2 * r20 + t0
    out1 = acc0 * r01 + acc1 * r11 + acc2 * r21 + t1
    out2 = acc0 * r02 + acc1 * r12 + acc2 * r22 + t2

    # ---------- Step 3: Write back the final global coordinates in one go ----------
    out_base = out_ptr + m * stride_out_m + p_offs * stride_out_p
    tl.store(out_base + 0, out0, mask=p_mask)
    tl.store(out_base + 1, out1, mask=p_mask)
    tl.store(out_base + 2, out2, mask=p_mask)


def triton_fused_linear_point_transform(x, weight_T, bias, rot_mats, trans, HP, allow_tf32=True):
    """
    Linear + Rigid transformation fusion external interface
    Parameters:
        x: [M, K] Flattened input features, M = B * N_res
        weight_T: [K, OUT_DIM] Transposed Linear weights, OUT_DIM = HP * 3
        bias: [OUT_DIM] Linear bias
        rot_mats: [M, 3, 3] Rotation matrices for each residue
        trans: [M, 3] Translation vectors for each residue
        HP: int Total number of points per residue = no_heads * num_points
        allow_tf32: bool Whether to enable TF32 tensor cores
    Returns:
        out: [M, HP, 3] Point coordinates in the global coordinate system
    """
    assert x.is_cuda and weight_T.is_cuda and bias.is_cuda
    assert rot_mats.is_cuda and trans.is_cuda
    assert x.shape[1] == weight_T.shape[0]
    assert weight_T.shape[1] == HP * 3
    assert rot_mats.shape[0] == x.shape[0] == trans.shape[0]
    assert rot_mats.shape[1:] == (3, 3) and trans.shape[1] == 3
    assert x.dtype == torch.float32

    M, K = x.shape
    out = torch.empty((M, HP, 3), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        M,
        triton.cdiv(HP, meta['BLOCK_P']),
    )

    fused_linear_point_transform_kernel[grid](
        x, weight_T, bias, rot_mats, trans, out,
        M, K, HP,
        x.stride(0), x.stride(1),
        weight_T.stride(0), weight_T.stride(1),
        rot_mats.stride(0),
        trans.stride(0),
        out.stride(0), out.stride(1),
        ALLOW_TF32=allow_tf32,
    )
    return out


# ==================== Attention Squared Distance Fusion Kernel  ====================
@triton.jit
def point_sq_dist_kernel(
    q_pts_ptr, k_pts_ptr, dist_ptr,
    N, P,
    stride_q_m, stride_q_n, stride_q_p, stride_q_c,
    stride_k_m, stride_k_n, stride_k_p, stride_k_c,
    stride_dist_m, stride_dist_n, stride_dist_j,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Each thread processes a row (one query position)
    row_idx = tl.program_id(0)
    head_idx = row_idx // N
    seq_i = row_idx % N

    n_offs = tl.arange(0, BLOCK_N)
    p_offs = tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    # Load the current row query point coordinates [P, 3], precompute q_sq (row-wise reuse)
    q_base = q_pts_ptr + head_idx * stride_q_m + seq_i * stride_q_n
    qp_x = tl.load(q_base + p_offs * stride_q_p + 0 * stride_q_c, mask=p_mask, other=0.0)
    qp_y = tl.load(q_base + p_offs * stride_q_p + 1 * stride_q_c, mask=p_mask, other=0.0)
    qp_z = tl.load(q_base + p_offs * stride_q_p + 2 * stride_q_c, mask=p_mask, other=0.0)
    q_sq = tl.sum(qp_x * qp_x + qp_y * qp_y + qp_z * qp_z, axis=0)

    k_head_base = k_pts_ptr + head_idx * stride_k_m
    dist_base = dist_ptr + head_idx * stride_dist_m + seq_i * stride_dist_n

    # Iterate over the key dimension in chunks and calculate the squared distance
    for j_start in range(0, N, BLOCK_N):
        j_offs = j_start + n_offs
        j_mask = j_offs < N

        kp_x = tl.load(
            k_head_base + j_offs[:, None] * stride_k_n + p_offs[None, :] * stride_k_p + 0 * stride_k_c,
            mask=j_mask[:, None] & p_mask[None, :], other=0.0
        )
        kp_y = tl.load(
            k_head_base + j_offs[:, None] * stride_k_n + p_offs[None, :] * stride_k_p + 1 * stride_k_c,
            mask=j_mask[:, None] & p_mask[None, :], other=0.0
        )
        kp_z = tl.load(
            k_head_base + j_offs[:, None] * stride_k_n + p_offs[None, :] * stride_k_p + 2 * stride_k_c,
            mask=j_mask[:, None] & p_mask[None, :], other=0.0
        )

        k_sq = tl.sum(kp_x * kp_x + kp_y * kp_y + kp_z * kp_z, axis=1)  # [BLOCK_N]
        qk_dot = (
            tl.sum(qp_x[None, :] * kp_x, axis=1) +
            tl.sum(qp_y[None, :] * kp_y, axis=1) +
            tl.sum(qp_z[None, :] * kp_z, axis=1)
        )  # [BLOCK_N]

        # Square distance formula:||q - k||² = q_sq + k_sq - 2*qk
        dist = q_sq + k_sq - 2.0 * qk_dot

        # Write back the result
        tl.store(dist_base + j_offs * stride_dist_j, dist, mask=j_mask)


def triton_point_sq_dist(q_pts, k_pts):
    """
    Point coordinate squared distance calculation
    Input shape: [B*H, N, P, 3]
    Output shape: [B*H, N, N]
    Equivalent to: (q_pts**2).sum(-1).sum(-1, keepdim=True) + (k_pts**2).sum(-1).sum(-2, keepdim=True) - 2 * einsum('b i p c, b j p c -> b i j', q_pts, k_pts)
    """
    assert q_pts.is_cuda and k_pts.is_cuda
    assert q_pts.dtype == torch.float32 and k_pts.dtype == torch.float32
    assert q_pts.shape[0] == k_pts.shape[0]
    assert q_pts.shape[2] == k_pts.shape[2]
    assert q_pts.shape[3] == 3 and k_pts.shape[3] == 3

    M, N_q, P, _ = q_pts.shape
    _, N_k, _, _ = k_pts.shape
    assert N_q == N_k
    N = N_q

    dist = torch.empty((M, N, N), device=q_pts.device, dtype=q_pts.dtype)

    # Automatic block parameter adaptation
    BLOCK_N = triton.next_power_of_2(min(N, 128))
    BLOCK_P = triton.next_power_of_2(P)
    grid = (M * N,)

    point_sq_dist_kernel[grid](
        q_pts, k_pts, dist,
        N, P,
        q_pts.stride(0), q_pts.stride(1), q_pts.stride(2), q_pts.stride(3),
        k_pts.stride(0), k_pts.stride(1), k_pts.stride(2), k_pts.stride(3),
        dist.stride(0), dist.stride(1), dist.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_P=BLOCK_P,
        num_warps=4
    )
    return dist