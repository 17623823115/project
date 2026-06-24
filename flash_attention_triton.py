import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
    ],
    key=['seq_len', 'BLOCK_D'],
)
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    blocks_per_head = tl.cdiv(seq_len, BLOCK_M)
    bh_id = pid // blocks_per_head  # 展平后的 batch+head 索引
    m_block_id = pid % blocks_per_head

    offs_m = m_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # 单个 head 的基地址，保证分块不跨 head
    head_base = bh_id * seq_len * BLOCK_D

    # 数值稳定变量初始化
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 编译期常量计算 scale，精度与 CUDA 对齐
    scale = 1.0 / tl.sqrt(float(BLOCK_D))

    # 加载 Q 分块
    q_ptrs = Q_ptr + head_base + offs_m[:, None] * BLOCK_D + offs_d[None, :]
    q_mask = offs_m[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0) * scale

    # 沿 K 维度分块主循环
    for start_n in range(0, seq_len, BLOCK_N):
        k_ptrs = K_ptr + head_base + (start_n + offs_n[:, None]) * BLOCK_D + offs_d[None, :]
        v_ptrs = V_ptr + head_base + (start_n + offs_n[:, None]) * BLOCK_D + offs_d[None, :]

        k_valid = (start_n + offs_n) < seq_len  # 形状 [BLOCK_N]
        k_mask_2d = k_valid[None, :]  # 与 qk 形状广播匹配

        k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)

        # 修复：强制纯 FP32 精度，与 CUDA 版本完全对齐
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")
        qk = tl.where(k_mask_2d, qk, -float('inf'))

        # 在线 Softmax 更新，数值稳定公式与 CUDA 完全一致
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = alpha * l_i + l_ij

        # 累加输出，同样强制 FP32 精度
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p, v, input_precision="ieee")

        m_i = m_new
        l_i = l_new

    # 归一化写回
    acc = acc / l_i[:, None]
    o_ptrs = O_ptr + head_base + offs_m[:, None] * BLOCK_D + offs_d[None, :]
    o_mask = offs_m[:, None] < seq_len
    tl.store(o_ptrs, acc, mask=o_mask)


def triton_flash_attention(Q, K, V):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dim() == 4  # [B, H, N, D]
    assert Q.dtype == torch.float32
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert Q.shape == K.shape == V.shape

    batch, heads, seq_len, head_dim = Q.shape
    O = torch.empty_like(Q)

    BLOCK_D = triton.next_power_of_2(head_dim)
    total_heads = batch * heads

    # 展平为 [B*H*N, D]，每个 head 的 query 连续排列
    Q_flat = Q.reshape(-1, head_dim)
    K_flat = K.reshape(-1, head_dim)
    V_flat = V.reshape(-1, head_dim)
    O_flat = O.reshape(-1, head_dim)

    grid = lambda meta: (
        total_heads * triton.cdiv(seq_len, meta['BLOCK_M']),
    )

    flash_attention_kernel[grid](
        Q_flat, K_flat, V_flat, O_flat,
        seq_len,
        BLOCK_D=BLOCK_D
    )
    return O_flat.reshape(batch, heads, seq_len, head_dim)