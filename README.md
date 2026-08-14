{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d594c0d9-10f1-4627-8d6c-1eecfda52553",
   "metadata": {},
   "source": [
    "A Systematic Performance Benchmark of Handwritten CUDA, Handwritten Triton and Native PyTorch Kernels\n",
    "\n",
    "Overview\n",
    "This repository contains the source code for the MSc research project: From Single Operators to End-to-End AlphaFold Structure Module. It implements three core deep learning operators (GEMM, LayerNorm, Softmax) in both handwritten native CUDA and handwritten Triton, and provides reproducible benchmark frameworks to compare their performance against the official PyTorch native implementation.\n",
    "\n",
    "The project includes two levels of evaluation:\n",
    "1. Single-kernel level: Isolated performance comparison of three operator implementations under controlled variables, in standard IEEE FP32 precision.\n",
    "2. End-to-end level: Full benchmark on the AlphaFold Structure Module, with kernel fusion optimization enabled for both CUDA and Triton implementations.\n",
    "\n",
    "All benchmarks use CUDA Events for high-precision kernel-level timing and take median latency as the final metric to ensure fair and reliable comparison.\n",
    "\n",
    "Prerequisites\n",
    "Hardware\n",
    "NVIDIA GPU with CUDA support (tested on NVIDIA GeForce RTX 5060 Laptop GPU, 8 GiB VRAM)\n",
    "Minimum 16 GiB system RAM\n",
    "\n",
    "Component\tVersion\n",
    "Python\t3.10.12\n",
    "System CUDA Toolkit\t13.3 (required for compiling CUDA extensions)\n",
    "PyTorch Bundled CUDA Runtime\t13.2\n",
    "PyTorch\t2.12.0\n",
    "Triton\t3.7.0\n",
    "Operating System\tUbuntu 22.04.5 LTS\n",
    "NVIDIA Driver\t592.01\n",
    "\n",
    "Project Structure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03c3af0e-2303-4d70-a2bb-94a00190b09b",
   "metadata": {},
   "outputs": [],
   "source": [
    ".\n",
    "├── profile/                      # Hardware profiling scripts for Nsight Compute\n",
    "│   ├── profile_gemm_native.py    \n",
    "│   ├── profile_gemm_cuda.py      \n",
    "│   ├── profile_gemm_triton.py    \n",
    "│   ├── profile_layernorm_native.py\n",
    "│   ├── profile_layernorm_cuda.py\n",
    "│   ├── profile_layernorm_triton.py\n",
    "│   ├── profile_softmax_native.py\n",
    "│   ├── profile_softmax_cuda.py\n",
    "│   └── profile_softmax_triton.py\n",
    "├── cuda_kernels.cu               # Handwritten CUDA kernel source code\n",
    "├── cuda_ops.py                   # Python wrapper for CUDA kernels\n",
    "├── layernorm__triton.py          # Triton implementation of LayerNorm\n",
    "├── triton_softmax.py             # Triton implementation of Softmax\n",
    "├── triton_kernels.py             # Triton implementation of GEMM and fused operators\n",
    "├── structure_module_standalone.py # Standalone AlphaFold Structure Module\n",
    "├── benchmark.py                  # Single-kernel three-way benchmark script\n",
    "├── benchmark_3way.py             # End-to-end Structure Module three-way benchmark\n",
    "├── setup.py                      # CUDA extension compilation configuration\n",
    "├── requirements.txt              # Python dependency list\n",
    "└── README.md"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a7648344-213a-4b21-a9e9-0eeb8b8b4720",
   "metadata": {},
   "source": [
    "Usage\n",
    "1. Single-Kernel Benchmark\n",
    "Runs full benchmark for LayerNorm, Softmax and GEMM operators, comparing handwritten CUDA, handwritten Triton and PyTorch native implementations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8c01e636-6804-4a11-bba6-648707534b62",
   "metadata": {},
   "outputs": [],
   "source": [
    "python benchmark.py"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e3226102-47cf-4625-992c-0d8937efaa77",
   "metadata": {},
   "source": [
    "Outputs median latency, speedup ratio and TFLOPS throughput for each test case. All tests are performed in pure FP32 precision with TF32 globally disabled."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54db29a0-a0d6-4141-a64a-0f9b5d7f5f65",
   "metadata": {},
   "source": [
    "2. End-to-End Structure Module Benchmark\n",
    "Runs end-to-end inference benchmark on the AlphaFold Structure Module across 5 sequence lengths (64 / 128 / 256 / 512 / 1024). Supports switching between native baseline, Triton-optimized and handwritten CUDA implementations, with kernel fusion optimization.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "346e30d8-bca5-47e1-9f7e-264912fb05fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "python benchmark_3way.py"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f48181e-4053-4e55-ab82-f6b6a09e34ef",
   "metadata": {},
   "source": [
    "Results include median latency, standard deviation, speedup ratio and FP32 computational throughput. Raw results are saved to 3way_benchmark.json.\n",
    "\n",
    "3. Hardware Profiling\n",
    "\n",
    "Each script in the profile/ directory is designed to work with NVIDIA Nsight Compute for collecting hardware-level metrics (SM utilization, DRAM bandwidth utilization, etc.).\n",
    "\n",
    "Example"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8db4ec1b-d54c-402d-8818-1934843c2572",
   "metadata": {},
   "outputs": [],
   "source": [
    "sudo -E env \"PATH=$PATH\" ncu \\\n",
    "  --section SpeedOfLight \\\n",
    "  -f \\\n",
    "  --launch-skip 20 \\\n",
    "  -o layernorm_native \\\n",
    "  python profile_layernorm_native.py"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8cddb396-8428-45bb-a15b-bdf0130d0952",
   "metadata": {},
   "source": [
    "get csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84692eb6-5472-43d8-b2f6-ba6db4a372e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "ncu --import layernorm_native.ncu-rep --csv > layernorm_native_metrics.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23e89d67-058e-4582-8945-a2a884ffb15f",
   "metadata": {},
   "source": [
    "Benchmark Methodology\n",
    "\n",
    "1. Timing: Kernel-level timing via torch.cuda.Event, which records timestamps directly at the GPU driver level and eliminates CPU scheduling overhead.\n",
    "2. Warm-up: All benchmarks include a dedicated warm-up phase to eliminate cold-start, compilation cache and GPU frequency scaling overhead.\n",
    "3. Statistical robustness: All results are reported as the median of multiple repeated runs to exclude system noise and outliers.\n",
    "4. Precision control: TF32 acceleration is globally disabled for both cuBLAS and cuDNN to ensure all calculations use standard IEEE FP32 precision.\n",
    "5. Correctness validation: All custom implementations are validated against PyTorch native output via torch.allclose with predefined tolerances.\n",
    "\n",
    "Notes\n",
    "1. All test inputs are randomly generated tensors with fixed seeds for reproducibility.\n",
    "2. Kernel fusion optimization is enabled by default in the Triton and CUDA modes of the end-to-end benchmark.\n",
    "3. For best consistency, set the GPU to fixed performance mode before running benchmarks to eliminate dynamic frequency scaling interference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c827ccd-2b54-4187-bf71-c7e177f047cb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
