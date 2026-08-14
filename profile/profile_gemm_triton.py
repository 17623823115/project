import torch
from triton_kernels import triton_linear_gemm

M = K = N = 2048
device = torch.device("cuda")

torch.manual_seed(42)
x = torch.randn(M, K, device=device, dtype=torch.float32).contiguous()
weight = torch.randn(N, K, device=device, dtype=torch.float32)
bias = torch.randn(N, device=device, dtype=torch.float32)
weight_T = weight.T.contiguous()


for _ in range(20):
    _ = triton_linear_gemm(x, weight_T, bias, relu=False, allow_tf32=False)
torch.cuda.synchronize()


for _ in range(100):
    out = triton_linear_gemm(x, weight_T, bias, relu=False, allow_tf32=False)
torch.cuda.synchronize()