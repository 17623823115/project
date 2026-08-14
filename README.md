A Systematic Performance Benchmark of Handwritten CUDA, Handwritten Triton and Native PyTorch Kernels

Overview
This repository contains the source code for the MSc research project: From Single Operators to End-to-End AlphaFold Structure Module. It implements three core deep learning operators (GEMM, LayerNorm, Softmax) in both handwritten native CUDA and handwritten Triton, and provides reproducible benchmark frameworks to compare their performance against the official PyTorch native implementation.

The project includes two levels of evaluation:
1. Single-kernel level: Isolated performance comparison of three operator implementations under controlled variables, in standard IEEE FP32 precision.
2. End-to-end level: Full benchmark on the AlphaFold Structure Module, with kernel fusion optimization enabled for both CUDA and Triton implementations.

All benchmarks use CUDA Events for high-precision kernel-level timing and take median latency as the final metric to ensure fair and reliable comparison.

Prerequisites
Hardware
NVIDIA GPU with CUDA support (tested on NVIDIA GeForce RTX 5060 Laptop GPU, 8 GiB VRAM)
Minimum 16 GiB system RAM

Component	Version
Python	3.10.12
System CUDA Toolkit	13.3 (required for compiling CUDA extensions)
PyTorch Bundled CUDA Runtime	13.2
PyTorch	2.12.0
Triton	3.7.0
Operating System	Ubuntu 22.04.5 LTS
NVIDIA Driver	592.01

Project Structure
.
├── profile/                      # Hardware profiling scripts for Nsight Compute
│   ├── profile_gemm_native.py    
│   ├── profile_gemm_cuda.py      
│   ├── profile_gemm_triton.py    
│   ├── profile_layernorm_native.py
│   ├── profile_layernorm_cuda.py
│   ├── profile_layernorm_triton.py
│   ├── profile_softmax_native.py
│   ├── profile_softmax_cuda.py
│   └── profile_softmax_triton.py
├── cuda_kernels.cu               # Handwritten CUDA kernel source code
├── cuda_ops.py                   # Python wrapper for CUDA kernels
├── layernorm__triton.py          # Triton implementation of LayerNorm
├── triton_softmax.py             # Triton implementation of Softmax
├── triton_kernels.py             # Triton implementation of GEMM and fused operators
├── structure_module_standalone.py # Standalone AlphaFold Structure Module
├── benchmark.py                  # Single-kernel three-way benchmark script
├── benchmark_3way.py             # End-to-end Structure Module three-way benchmark
├── setup.py                      # CUDA extension compilation configuration
├── requirements.txt              # Python dependency list
└── README.md

Usage
1. Single-Kernel Benchmark
Runs full benchmark for LayerNorm, Softmax and GEMM operators, comparing handwritten CUDA, handwritten Triton and PyTorch native implementations.

python benchmark.py

Outputs median latency, speedup ratio and TFLOPS throughput for each test case. All tests are performed in pure FP32 precision with TF32 globally disabled.

2. End-to-End Structure Module Benchmark
Runs end-to-end inference benchmark on the AlphaFold Structure Module across 5 sequence lengths (64 / 128 / 256 / 512 / 1024). Supports switching between native baseline, Triton-optimized and handwritten CUDA implementations, with kernel fusion optimization.

python benchmark_3way.py

Results include median latency, standard deviation, speedup ratio and FP32 computational throughput. Raw results are saved to 3way_benchmark.json.

3. Hardware Profiling

Each script in the profile/ directory is designed to work with NVIDIA Nsight Compute for collecting hardware-level metrics (SM utilization, DRAM bandwidth utilization, etc.).

Example

sudo -E env "PATH=$PATH" ncu \
  --section SpeedOfLight \
  -f \
  --launch-skip 20 \
  -o layernorm_native \
  python profile_layernorm_native.py

ncu --import layernorm_native.ncu-rep --csv > layernorm_native_metrics.csv

Benchmark Methodology

1. Timing: Kernel-level timing via torch.cuda.Event, which records timestamps directly at the GPU driver level and eliminates CPU scheduling overhead.
2. Warm-up: All benchmarks include a dedicated warm-up phase to eliminate cold-start, compilation cache and GPU frequency scaling overhead.
3. Statistical robustness: All results are reported as the median of multiple repeated runs to exclude system noise and outliers.
4. Precision control: TF32 acceleration is globally disabled for both cuBLAS and cuDNN to ensure all calculations use standard IEEE FP32 precision.
5. Correctness validation: All custom implementations are validated against PyTorch native output via torch.allclose with predefined tolerances.

Notes
1. All test inputs are randomly generated tensors with fixed seeds for reproducibility.
2. Kernel fusion optimization is enabled by default in the Triton and CUDA modes of the end-to-end benchmark.
3. For best consistency, set the GPU to fixed performance mode before running benchmarks to eliminate dynamic frequency scaling interference.