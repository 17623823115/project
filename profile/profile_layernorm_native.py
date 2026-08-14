import torch
from torch import nn


M, C = 4096, 384
EPS = 1e-5
device = torch.device("cuda")

torch.manual_seed(42)
x = torch.randn(M, C, device=device, dtype=torch.float32).contiguous()
weight = torch.randn(C, device=device, dtype=torch.float32)
bias = torch.randn(C, device=device, dtype=torch.float32)


native_ln = nn.LayerNorm(C, eps=EPS).to(device)
native_ln.weight.data = weight
native_ln.bias.data = bias

for _ in range(20):
    _ = native_ln(x)
torch.cuda.synchronize()

for _ in range(100):
    out = native_ln(x)
torch.cuda.synchronize()