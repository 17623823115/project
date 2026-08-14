import torch
import json
from structure_module_standalone import (
    StructureModule, Linear, LayerNorm, 
    InvariantPointAttention, Softmax
)

CONFIG = {
    "BATCH_SIZE": 1,
    "N_RES_LIST": [64, 128, 256, 512, 1024],
    "C_S": 384,
    "C_Z": 128,
    "DTYPE": torch.float32,
    "DEVICE": "cuda",
    "WARMUP_ROUNDS": 50,
    "TEST_ROUNDS": 200,
    "RTOL": 5e-3,
    "ATOL": 3e-3,
    "SAVE_LOG": True,
    "LOG_PATH": "3way_benchmark.json",
    "STRUCTURE_CONFIG": {
        "c_s": 384, "c_z": 128, "c_ipa": 16, "c_resnet": 128,
        "no_heads_ipa": 12, "no_qk_points": 4, "no_v_points": 8,
        "dropout_rate": 0.1, "no_blocks": 8, "no_transition_layers": 1,
        "no_resnet_blocks": 2, "no_angles": 7, "trans_scale_factor": 10.0,
        "epsilon": 1e-8, "inf": 1e5, "is_multimer": False
    }
}

def print_env_info():
    print("=" * 80)
    print("Three-way Performance Comparison: Native PyTorch vs Triton-Optimized vs Handwritten CUDA")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"GPU Model: {torch.cuda.get_device_name()}")
    print(f"Data Precision: {CONFIG['DTYPE']}")
    print("=" * 80)

def gen_fixed_input(n_res, seed=42):
    torch.manual_seed(seed)
    B = CONFIG["BATCH_SIZE"]
    c_s, c_z = CONFIG["C_S"], CONFIG["C_Z"]
    dev, dtype = CONFIG["DEVICE"], CONFIG["DTYPE"]
    evo = {
        "single": torch.randn(B, n_res, c_s, device=dev, dtype=dtype).contiguous(),
        "pair": torch.randn(B, n_res, n_res, c_z, device=dev, dtype=dtype).contiguous()
    }
    aatype = torch.randint(0, 21, (B, n_res), device=dev)
    mask = torch.ones(B, n_res, device=dev, dtype=dtype)
    return evo, aatype, mask

def benchmark_model(model, evo, aatype, mask):
    warmup = CONFIG["WARMUP_ROUNDS"]
    repeat = CONFIG["TEST_ROUNDS"]
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(evo, aatype, mask)
        torch.cuda.synchronize()
    
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    timings = []
    with torch.no_grad():
        for _ in range(repeat):
            start_ev.record()
            _ = model(evo, aatype, mask)
            end_ev.record()
            torch.cuda.synchronize()
            timings.append(start_ev.elapsed_time(end_ev))
    
    timings_tensor = torch.tensor(timings)
    return timings_tensor.median().item(), timings_tensor.std().item()

def calculate_structure_module_flops(n_res, config):
    """Calculate total theoretical floating-point operations for a full Structure Module forward pass (unit: FLOP)"""
    c_s = config["c_s"]
    c_z = config["c_z"]
    c_ipa = config["c_ipa"]
    no_heads = config["no_heads_ipa"]
    no_qk_points = config["no_qk_points"]
    no_v_points = config["no_v_points"]
    no_blocks = config["no_blocks"]
    
    flops_per_block = 0
    
    # 1. IPA scalar Linear layers: q / kv / b / out
    flops_per_block += 2 * n_res * c_s * no_heads * c_ipa  # q projection
    flops_per_block += 2 * 2 * n_res * c_s * no_heads * c_ipa  # kv projection (k+v paths)
    flops_per_block += 2 * n_res * n_res * c_z * no_heads  # pair feature b projection
    out_dim = no_heads * (c_ipa + c_z + 4 * no_v_points)
    flops_per_block += 2 * n_res * out_dim * c_s  # output projection
    
    # 2. IPA attention QK dot product
    flops_per_block += 2 * no_heads * n_res * n_res * c_ipa
    
    # 3. Point coordinate projection Linear layers
    total_points = no_qk_points + 2 * no_v_points
    flops_per_block += 2 * n_res * c_s * no_heads * total_points * 3
    
    # 4. Transition layer double Linear operations
    flops_per_block += 2 * 2 * n_res * c_s * c_s
    
    # 5. Lightweight operations (norm, activation, residual, ~5% of total)
    flops_per_block *= 1.05
    
    # Total FLOPs = per-block × number of blocks
    total_flops = flops_per_block * no_blocks
    return total_flops

def set_mode(mode):
    """Unified mode switch: native / triton / cuda"""
    if mode == "native":
        Linear.use_triton = False
        Linear.use_cuda = False
        LayerNorm.use_triton = False
        LayerNorm.use_cuda = False
        Softmax.use_triton = False
        InvariantPointAttention.use_fused_attention = False
    elif mode == "triton":
        Linear.use_triton = True
        Linear.use_cuda = False
        LayerNorm.use_triton = True
        LayerNorm.use_cuda = False
        Softmax.use_triton = True
        InvariantPointAttention.use_fused_attention = True
    elif mode == "cuda":
        Linear.use_triton = False
        Linear.use_cuda = True
        LayerNorm.use_triton = False
        LayerNorm.use_cuda = True
        Softmax.use_triton = False
        InvariantPointAttention.use_fused_attention = True

def main():
    if CONFIG["DEVICE"] == "cuda":
        torch.cuda.empty_cache()
    print_env_info()
    
    torch.manual_seed(42)
    model = StructureModule(**CONFIG["STRUCTURE_CONFIG"]).to(
        CONFIG["DEVICE"], dtype=CONFIG["DTYPE"]
    )
    model.eval()
    results = []

    # Print table header with TFLOPS column
    print(f"{'Size':<10} | {'Impl':<12} | {'Median Lat (ms)':<14} | {'Std Dev (ms)':<12} | {'Speedup':<8} | {'TFLOPS':<8}")
    print("-" * 90)

    for n_res in CONFIG["N_RES_LIST"]:
        evo, aatype, mask = gen_fixed_input(n_res)
        
        # 1. Native baseline
        set_mode("native")
        ref_med, ref_std = benchmark_model(model, evo, aatype, mask)
        # 2. Triton-optimized version
        set_mode("triton")
        tri_med, tri_std = benchmark_model(model, evo, aatype, mask)
        # 3. Handwritten CUDA version
        set_mode("cuda")
        cuda_med, cuda_std = benchmark_model(model, evo, aatype, mask)

        # ========== Added: calculate theoretical operations and throughput for current sequence length ==========
        total_flops = calculate_structure_module_flops(n_res, CONFIG["STRUCTURE_CONFIG"])
        ref_tflops = total_flops / (ref_med / 1000) / 1e12
        tri_tflops = total_flops / (tri_med / 1000) / 1e12
        cuda_tflops = total_flops / (cuda_med / 1000) / 1e12

        # Print results (added TFLOPS column)
        print(f"[{n_res:<5}] | Native PyTorch | {ref_med:<14.4f} | {ref_std:<12.4f} | {1.00:<8.2f} | {ref_tflops:<8.2f}")
        print(f"[{n_res:<5}] | Triton Opt    | {tri_med:<14.4f} | {tri_std:<12.4f} | {ref_med/tri_med:<8.2f} | {tri_tflops:<8.2f}")
        print(f"[{n_res:<5}] | Handwritten CUDA | {cuda_med:<14.4f} | {cuda_std:<12.4f} | {ref_med/cuda_med:<8.2f} | {cuda_tflops:<8.2f}")
        print()

        # Store results in dict (added TFLOPS field)
        results.append({
            "n_res": n_res,
            "native": {"median_ms": ref_med, "std_ms": ref_std, "tflops": ref_tflops},
            "triton": {"median_ms": tri_med, "std_ms": tri_std, "tflops": tri_tflops},
            "cuda": {"median_ms": cuda_med, "std_ms": cuda_std, "tflops": cuda_tflops}
        })

    if CONFIG["SAVE_LOG"]:
        with open(CONFIG["LOG_PATH"], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Experiment log saved to {CONFIG['LOG_PATH']}")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Linear.use_tf32 = False
    main()