import torch
from layernorm__triton import triton_layernorm

M, C = 4096, 384
EPS = 1e-5
device = torch.device("cuda")

torch.manual_seed(42)
x = torch.randn(M, C, device=device, dtype=torch.float32).contiguous()
weight = torch.randn(C, device=device, dtype=torch.float32)
bias = torch.randn(C, device=device, dtype=torch.float32)


for _ in range(20):
    _ = triton_layernorm(x, weight, bias, EPS)
torch.cuda.synchronize()


for _ in range(100):
    out = triton_layernorm(x, weight, bias, EPS)
torch.cuda.synchronize()