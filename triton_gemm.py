import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel_v2_fixed(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    # ========== 修复：修正分组PID计算逻辑，缩进统一 ==========
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # ========== 修复：移除dtype参数，兼容低版本Triton ==========
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 指针计算（移除手动对齐逻辑，Triton自动处理对齐）
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # FP32累加器，与CUDA精度对齐
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ========== 修复：移除tl.unroll，兼容旧版本 ==========
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # 强制纯FP32 IEEE精度，与CUDA完全对齐
        acc += tl.dot(a, b, input_precision="ieee")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回结果
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def your_gemm(A, B):
    assert A.is_cuda and B.is_cuda
    assert A.shape[1] == B.shape[0]
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )
    gemm_kernel_v2_fixed[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# 自带测试
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda")
    M, N, K = 1024, 1024, 1024

    A = torch.randn(M, K, device=device, dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device=device, dtype=torch.float32).contiguous()

    ref_C = torch.matmul(A, B)
    test_C = your_gemm(A, B)
    
    rtol, atol = 1e-3, 1e-4
    is_correct = torch.allclose(test_C, ref_C, rtol=rtol, atol=atol)
    max_error = (test_C - ref_C).abs().max().item()
    print(f"功能测试: {'PASS' if is_correct else 'FAIL'} | 最大误差: {max_error:.6e}")

    # 性能测试
    warmup = 20
    repeat = 100
    for _ in range(warmup):
        _ = your_gemm(A, B)
    
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(repeat):
        start_ev.record()
        _ = your_gemm(A, B)
        end_ev.record()
        torch.cuda.synchronize()
        timings.append(start_ev.elapsed_time(end_ev))
    
    median_ms = torch.tensor(timings).median().item()
    flops = 2.0 * M * N * K
    gflops = flops / (median_ms / 1000) / 1e9
    print(f"Triton修正版: 中位延迟 {median_ms:.4f} ms | 算力 {gflops:.2f} GFLOPS")