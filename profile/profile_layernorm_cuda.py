import torch
from cuda_ops import cuda_layernorm


M, C = 4096, 384
EPS = 1e-5
device = torch.device("cuda")

torch.manual_seed(42)
x = torch.randn(M, C, device=device, dtype=torch.float32).contiguous()
weight = torch.randn(C, device=device, dtype=torch.float32)
bias = torch.randn(C, device=device, dtype=torch.float32)


for _ in range(20):
    _ = cuda_layernorm(x, weight, bias, EPS)
torch.cuda.synchronize()


for _ in range(100):
    out = cuda_layernorm(x, weight, bias, EPS)
torch.cuda.synchronize()