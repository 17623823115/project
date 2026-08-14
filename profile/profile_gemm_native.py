import torch
from torch import nn


M = K = N = 2048
device = torch.device("cuda")


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(42)
x = torch.randn(M, K, device=device, dtype=torch.float32).contiguous()
weight = torch.randn(N, K, device=device, dtype=torch.float32)
bias = torch.randn(N, device=device, dtype=torch.float32)

native_linear = nn.Linear(K, N, bias=True).to(device)
native_linear.weight.data = weight
native_linear.bias.data = bias


for _ in range(20):
    _ = native_linear(x)
torch.cuda.synchronize()

for _ in range(100):
    out = native_linear(x)
torch.cuda.synchronize()