import torch
import torch.nn as nn
import triton
# 1. Import handwritten CUDA extension
import cuda_kernels
# 2. Import custom Triton operators
from layernorm__triton import triton_layernorm
from triton_softmax import _softmax_row_kernel
from triton_kernels import triton_linear_gemm

# -------------------------- Generic Timing Utility (Median Latency) --------------------------
def benchmark_fn(fn, *args, n_warmup=20, n_iter=100, **kwargs) -> float:
    """High-precision timing via CUDA Events, returns median latency of a single operator in milliseconds.
    Records latency per call and takes the median to eliminate outlier jitter for more robust results.
    """
    # Warm-up phase to eliminate cold-start and compilation cache overhead
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Formal test, records per-call latency
    time_list = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end))
    
    # Compute median
    time_list.sort()
    mid = n_iter // 2
    if n_iter % 2 == 0:
        median_ms = (time_list[mid - 1] + time_list[mid]) / 2
    else:
        median_ms = time_list[mid]
    return median_ms

def triton_softmax_wrapper(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for Triton Softmax, reuses the original tiling and warp configuration"""
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    M = x.numel() // x.size(-1)
    N = x.size(-1)
    output = torch.empty_like(x)
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
    _softmax_row_kernel[grid](
        x, output, M, N,
        x.stride(-2),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps
    )
    return output

# -------------------------- 1. LayerNorm Benchmark (Official Baseline: nn.LayerNorm) --------------------------
def test_layernorm(M: int, C: int, eps: float = 1e-5, n_warmup=20, n_iter=100):
    print(f"\n{'='*60}")
    print(f"LayerNorm Benchmark | Shape: [{M}, {C}] | eps={eps} | Official baseline: nn.LayerNorm | Metric: median latency")
    print(f"{'='*60}")
    
    x = torch.randn(M, C, device='cuda', dtype=torch.float32)
    gamma = torch.randn(C, device='cuda', dtype=torch.float32)
    beta = torch.randn(C, device='cuda', dtype=torch.float32)

    # 1. Official PyTorch: nn.LayerNorm
    layer_norm = nn.LayerNorm(C, eps=eps, device='cuda', dtype=torch.float32)
    # Replace with generated weights to ensure consistent input
    with torch.no_grad():
        layer_norm.weight.copy_(gamma)
        layer_norm.bias.copy_(beta)
    
    def torch_fn():
        return layer_norm(x)
    torch_out = torch_fn()
    torch_time = benchmark_fn(torch_fn, n_warmup=n_warmup, n_iter=n_iter)

    # 2. Handwritten CUDA
    def cuda_fn():
        if C == 128:
            return cuda_kernels.cuda_layernorm_128(x, gamma, beta, eps)
        elif C == 384:
            return cuda_kernels.cuda_layernorm_384(x, gamma, beta, eps)
        else:
            return cuda_kernels.cuda_layernorm_generic(x, gamma, beta, eps)
    cuda_out = cuda_fn()
    cuda_time = benchmark_fn(cuda_fn, n_warmup=n_warmup, n_iter=n_iter)

    # 3. Triton
    def triton_fn():
        return triton_layernorm(x, gamma, beta, eps)
    triton_out = triton_fn()
    triton_time = benchmark_fn(triton_fn, n_warmup=n_warmup, n_iter=n_iter)

    # Correctness validation
    correct_cuda = torch.allclose(cuda_out, torch_out, rtol=1e-3, atol=1e-3)
    correct_triton = torch.allclose(triton_out, torch_out, rtol=1e-3, atol=1e-3)
    print(f"PyTorch nn.LayerNorm: {torch_time:.4f} ms")
    print(f"Handwritten CUDA:     {cuda_time:.4f} ms | Correctness: {' Pass' if correct_cuda else ' Fail'}")
    print(f"Triton:               {triton_time:.4f} ms | Correctness: {' Pass' if correct_triton else ' Fail'}")
    print(f"Speedup (CUDA vs Official):   {torch_time / cuda_time:.2f}x")
    print(f"Speedup (Triton vs Official): {torch_time / triton_time:.2f}x")
    print(f"Speedup (Triton vs CUDA):     {cuda_time / triton_time:.2f}x")

# -------------------------- 2. Softmax Benchmark (Official Baseline: nn.Softmax) --------------------------
def test_softmax(M: int, N: int, n_warmup=20, n_iter=100):
    print(f"\n{'='*60}")
    print(f"Softmax Benchmark | Shape: [{M}, {N}] | dim=-1 | Official baseline: nn.Softmax | Metric: median latency")
    print(f"{'='*60}")
    
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)

    # 1. Official PyTorch: nn.Softmax
    softmax_layer = nn.Softmax(dim=-1)
    
    def torch_fn():
        return softmax_layer(x)
    torch_out = torch_fn()
    torch_time = benchmark_fn(torch_fn, n_warmup=n_warmup, n_iter=n_iter)

    # 2. Handwritten CUDA
    def cuda_fn():
        return cuda_kernels.cuda_softmax(x)
    cuda_out = cuda_fn()
    cuda_time = benchmark_fn(cuda_fn, n_warmup=n_warmup, n_iter=n_iter)

    # 3. Triton
    def triton_fn():
        return triton_softmax_wrapper(x)
    triton_out = triton_fn()
    triton_time = benchmark_fn(triton_fn, n_warmup=n_warmup, n_iter=n_iter)

    # Correctness validation
    correct_cuda = torch.allclose(cuda_out, torch_out, rtol=1e-3, atol=1e-3)
    correct_triton = torch.allclose(triton_out, torch_out, rtol=1e-3, atol=1e-3)
    print(f"PyTorch nn.Softmax: {torch_time:.4f} ms")
    print(f"Handwritten CUDA:   {cuda_time:.4f} ms | Correctness: {' Pass' if correct_cuda else ' Fail'}")
    print(f"Triton:             {triton_time:.4f} ms | Correctness: {' Pass' if correct_triton else ' Fail'}")
    print(f"Speedup (CUDA vs Official):   {torch_time / cuda_time:.2f}x")
    print(f"Speedup (Triton vs Official): {torch_time / triton_time:.2f}x")
    print(f"Speedup (Triton vs CUDA):     {cuda_time / triton_time:.2f}x")

# -------------------------- 3. GEMM Benchmark (Official Baseline: nn.Linear) --------------------------
def test_gemm(M: int, K: int, N: int, has_relu: bool = False, n_warmup=20, n_iter=100):
    print(f"\n{'='*60}")
    print(f"GEMM Benchmark | M={M}, K={K}, N={N} | ReLU={has_relu} | Official baseline: nn.Linear | Pure FP32 | Metric: median latency")
    print(f"{'='*60}")
    
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    # nn.Linear weight shape: [out_features, in_features] = [N, K]
    # Computation: y = x @ weight.T + bias
    weight = torch.randn(N, K, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)

    # 1. Official PyTorch: nn.Linear + optional ReLU
    linear_layer = nn.Linear(K, N, bias=True, device='cuda', dtype=torch.float32)
    # Replace with generated weights to ensure consistent input
    with torch.no_grad():
        linear_layer.weight.copy_(weight)
        linear_layer.bias.copy_(bias)
    
    relu_layer = nn.ReLU()
    
    def torch_fn():
        out = linear_layer(x)
        if has_relu:
            out = relu_layer(out)
        return out
    torch_out = torch_fn()
    torch_time = benchmark_fn(torch_fn, n_warmup=n_warmup, n_iter=n_iter)

    # Custom operators use weight.T as matrix B to maintain mathematical equivalence
    B = weight.T.contiguous()

    # 2. Handwritten CUDA
    def cuda_fn():
        return cuda_kernels.cuda_gemm_bias_relu(x, B, bias, has_relu)
    cuda_out = cuda_fn()
    cuda_time = benchmark_fn(cuda_fn, n_warmup=n_warmup, n_iter=n_iter)

    # 3. Triton (TF32 disabled, pure FP32 general cores)
    def triton_fn():
        return triton_linear_gemm(x, B, bias, relu=has_relu, allow_tf32=False)
    triton_out = triton_fn()
    triton_time = benchmark_fn(triton_fn, n_warmup=n_warmup, n_iter=n_iter)

    # Correctness validation
    correct_cuda = torch.allclose(cuda_out, torch_out, rtol=1e-3, atol=1e-3)
    correct_triton = torch.allclose(triton_out, torch_out, rtol=1e-3, atol=1e-3)

    # Calculate theoretical compute throughput
    flops = 2 * M * K * N
    torch_tflops = flops / (torch_time * 1e-3) / 1e12
    cuda_tflops = flops / (cuda_time * 1e-3) / 1e12
    triton_tflops = flops / (triton_time * 1e-3) / 1e12

    print(f"PyTorch nn.Linear: {torch_time:.4f} ms | Throughput: {torch_tflops:.2f} TFLOPS")
    print(f"Handwritten CUDA:  {cuda_time:.4f} ms | Throughput: {cuda_tflops:.2f} TFLOPS | Correctness: {' Pass' if correct_cuda else ' Fail'}")
    print(f"Triton:            {triton_time:.4f} ms | Throughput: {triton_tflops:.2f} TFLOPS | Correctness: {' Pass' if correct_triton else ' Fail'}")
    print(f"Speedup (CUDA vs Official):   {torch_time / cuda_time:.2f}x")
    print(f"Speedup (Triton vs Official): {torch_time / triton_time:.2f}x")
    print(f"Speedup (Triton vs CUDA):     {cuda_time / triton_time:.2f}x")

# -------------------------- 4. Pure GEMM Benchmark (No Bias) | Official Baseline: nn.Linear(bias=False) --------------------------
def test_gemm_no_bias(M: int, K: int, N: int, n_warmup=20, n_iter=100):
    print(f"\n{'='*60}")
    print(f"Pure GEMM Benchmark (No Bias) | M={M}, K={K}, N={N} | Official baseline: nn.Linear(bias=False) | Pure FP32 | Metric: median latency")
    print(f"{'='*60}")
    
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    # nn.Linear weight shape: [out_features, in_features] = [N, K]
    # Computation: y = x @ weight.T (no bias)
    weight = torch.randn(N, K, device='cuda', dtype=torch.float32)

    # 1. Official PyTorch: nn.Linear(bias=False)
    linear_layer = nn.Linear(K, N, bias=False, device='cuda', dtype=torch.float32)
    # Replace with generated weights to ensure consistent input
    with torch.no_grad():
        linear_layer.weight.copy_(weight)
    
    def torch_fn():
        return linear_layer(x)
    torch_out = torch_fn()
    torch_time = benchmark_fn(torch_fn, n_warmup=n_warmup, n_iter=n_iter)

    # Custom operators use weight.T as matrix B to maintain mathematical equivalence
    B = weight.T.contiguous()
    # Pass zero bias, mathematically equivalent to pure GEMM; performance overhead of adding zero is negligible
    zero_bias = torch.zeros(N, device='cuda', dtype=torch.float32)

    # 2. Handwritten CUDA
    def cuda_fn():
        return cuda_kernels.cuda_gemm_bias_relu(x, B, zero_bias, False)
    cuda_out = cuda_fn()
    cuda_time = benchmark_fn(cuda_fn, n_warmup=n_warmup, n_iter=n_iter)

    # 3. Triton (TF32 disabled, pure FP32 general cores)
    def triton_fn():
        return triton_linear_gemm(x, B, zero_bias, relu=False, allow_tf32=False)
    triton_out = triton_fn()
    triton_time = benchmark_fn(triton_fn, n_warmup=n_warmup, n_iter=n_iter)

    # Correctness validation
    correct_cuda = torch.allclose(cuda_out, torch_out, rtol=1e-3, atol=1e-3)
    correct_triton = torch.allclose(triton_out, torch_out, rtol=1e-3, atol=1e-3)

    # Calculate theoretical compute throughput
    flops = 2 * M * K * N
    torch_tflops = flops / (torch_time * 1e-3) / 1e12
    cuda_tflops = flops / (cuda_time * 1e-3) / 1e12
    triton_tflops = flops / (triton_time * 1e-3) / 1e12

    print(f"PyTorch nn.Linear (No Bias): {torch_time:.4f} ms | Throughput: {torch_tflops:.2f} TFLOPS")
    print(f"Handwritten CUDA:            {cuda_time:.4f} ms | Throughput: {cuda_tflops:.2f} TFLOPS | Correctness: {' Pass' if correct_cuda else ' Fail'}")
    print(f"Triton:                      {triton_time:.4f} ms | Throughput: {triton_tflops:.2f} TFLOPS | Correctness: {' Pass' if correct_triton else ' Fail'}")
    print(f"Speedup (CUDA vs Official):   {torch_time / cuda_time:.2f}x")
    print(f"Speedup (Triton vs Official): {torch_time / triton_time:.2f}x")
    print(f"Speedup (Triton vs CUDA):     {cuda_time / triton_time:.2f}x")
    
# -------------------------- Main Entry --------------------------
if __name__ == "__main__":
    # Globally disable TF32 for fair pure FP32 comparison
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("=" * 60)
    print("Handwritten CUDA vs Triton vs PyTorch Single-Operator Performance Benchmark")
    print("Official baseline uniformly uses torch.nn interface | Precision: standard IEEE FP32 | Metric: median latency")
    print("=" * 60)

    # ========== Test cases, can be modified as needed ==========
    layernorm_cases = [
        (4096, 128),
        (4096, 384),
        (4096, 512),
        (4096, 1024),
    ]
    for M, C in layernorm_cases:
        test_layernorm(M, C)

    softmax_cases = [
        (4096, 128),
        (4096, 512),
        (4096, 1024),
        (1024, 2048),
    ]
    for M, N in softmax_cases:
        test_softmax(M, N)

    gemm_cases = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 1024, 2048),
        (4096, 4096, 4096),
    ]
    for M, K, N in gemm_cases:
        test_gemm(M, K, N, has_relu=False)
    
    print("\n" + "=" * 60)
    print("=== No-Bias Pure GEMM Tests ===")
    print("=" * 60)
    
    gemm_no_bias_cases = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 1024, 2048),
        (4096, 4096, 4096),
    ]
    for M, K, N in gemm_no_bias_cases:
        test_gemm_no_bias(M, K, N)
    
    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)