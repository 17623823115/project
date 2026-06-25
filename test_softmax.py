import torch
from torch.utils.cpp_extension import load

# 加载CUDA实现
softmax_cuda_module = load(
    name="softmax_cuda",
    sources=["softmax_cuda.cu"],
    extra_cuda_cflags=["-O3", "-lineinfo"]
)
cuda_softmax = softmax_cuda_module.cuda_softmax

# 加载Triton实现
from softmax_triton import triton_softmax

# 官方参考实现
def ref_softmax(x):
    return torch.nn.functional.softmax(x, dim=-1)

# ===================== 工具函数 =====================
def print_env_info():
    """打印测试环境信息，保证实验可复现"""
    print("=" * 70)
    print("测试环境信息")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU型号: {torch.cuda.get_device_name()}")
    print(f"GPU算力: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"TF32禁用状态: matmul={not torch.backends.cuda.matmul.allow_tf32}")
    print("=" * 70)

def analyze_error(pred, ref):
    """详细误差分析：最大误差、平均误差"""
    err = (pred - ref).abs()
    return {
        "max": err.max().item(),
        "mean": err.mean().item()
    }

# ==================================================
# 1. 功能正确性测试
# ==================================================
def test_correctness():
    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(42)

    # 扩充测试尺寸：覆盖极端边界、对齐/非对齐、大小维度全场景
    test_sizes = [
        (1, 1),        # 极端最小尺寸
        (128, 128),    # 小尺寸对齐
        (256, 512),    # 中小尺寸
        (33, 97),      # 非对齐边界尺寸
        (1, 4096),     # 单行长序列
        (1024, 2048),  # 中等尺寸
        (4096, 64),    # 多行小维度
        (4096, 4096),  # 大尺寸对齐
        (8192, 8192),  # 超大尺寸
        (1023, 99),    # 极端非对齐
    ]

    all_pass = True
    rtol, atol = 1e-3, 1e-4
    print("========== Softmax 功能测试 ==========")

    for rows, cols in test_sizes:
        x = torch.randn(rows, cols, device=device, dtype=dtype).contiguous()
        
        ref_out = ref_softmax(x)
        cuda_out = cuda_softmax(x)
        triton_out = triton_softmax(x)

        cuda_correct = torch.allclose(cuda_out, ref_out, rtol=rtol, atol=atol)
        triton_correct = torch.allclose(triton_out, ref_out, rtol=rtol, atol=atol)
        
        cuda_err = analyze_error(cuda_out, ref_out)
        triton_err = analyze_error(triton_out, ref_out)

        size_str = f"[{rows:4d}, {cols:4d}]"
        print(f"{size_str} | CUDA:   {'PASS' if cuda_correct else 'FAIL'} | 最大误差: {cuda_err['max']:.6e} | 平均误差: {cuda_err['mean']:.6e}")
        print(f"{size_str} | Triton: {'PASS' if triton_correct else 'FAIL'} | 最大误差: {triton_err['max']:.6e} | 平均误差: {triton_err['mean']:.6e}")
        print()
        
        if not (cuda_correct and triton_correct):
            all_pass = False

    print(f"功能测试总结果：{'全部通过 ' if all_pass else '存在失败 '}")
    return all_pass

# ==================================================
# 2. 性能基准测试
# ==================================================
def benchmark_kernel(kernel_func, x, warmup=30, repeat=150):
    """基准测试：返回中位延迟、延迟标准差，提升统计稳定性"""
    for _ in range(warmup):
        _ = kernel_func(x)
    
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(repeat):
        start_ev.record()
        _ = kernel_func(x)
        end_ev.record()
        torch.cuda.synchronize()
        timings.append(start_ev.elapsed_time(end_ev))
    
    timings_tensor = torch.tensor(timings)
    median_ms = timings_tensor.median().item()
    std_ms = timings_tensor.std().item()
    return median_ms, std_ms

def test_performance():
    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(42)

    # 多尺寸性能扫描，覆盖不同量级和场景
    perf_sizes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (1, 16384),
        (16384, 64),
        (1023, 99)
    ]

    print("\n========== Softmax 性能测试 ==========")
    print(f"{'尺寸':<16} | {'实现':<10} | {'中位延迟(ms)':<12} | {'标准差(ms)':<10} | {'吞吐量(Gelem/s)':<16} | {'相对官方加速比':<12}")
    print("-" * 95)

    for rows, cols in perf_sizes:
        x = torch.randn(rows, cols, device=device, dtype=dtype).contiguous()
        total_elems = rows * cols

        cuda_ms, cuda_std = benchmark_kernel(cuda_softmax, x)
        triton_ms, triton_std = benchmark_kernel(triton_softmax, x)
        ref_ms, ref_std = benchmark_kernel(ref_softmax, x)

        # 计算吞吐量：每秒处理的元素数
        cuda_tp = total_elems / (cuda_ms / 1000) / 1e9
        triton_tp = total_elems / (triton_ms / 1000) / 1e9
        ref_tp = total_elems / (ref_ms / 1000) / 1e9

        size_str = f"[{rows}, {cols}]"
        print(f"{size_str:<16} | CUDA       | {cuda_ms:<12.4f} | {cuda_std:<10.4f} | {cuda_tp:<16.2f} | {ref_ms/cuda_ms:<10.2f}x")
        print(f"{'':<16} | Triton     | {triton_ms:<12.4f} | {triton_std:<10.4f} | {triton_tp:<16.2f} | {ref_ms/triton_ms:<10.2f}x")
        print(f"{'':<16} | PyTorch官方 | {ref_ms:<12.4f} | {ref_std:<10.4f} | {ref_tp:<16.2f} | 1.00x")
        print(f"{'':<16} | Triton/CUDA加速比: {cuda_ms / triton_ms:.2f}x")
        print()

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    print_env_info()
    if test_correctness():
        test_performance()
    else:
        print("\n功能测试未通过，请先修正内核正确性，再进行性能测试。")