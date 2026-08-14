import torch
from torch import nn

B, H, N_RES = 1, 12, 512
device = torch.device("cuda")

torch.manual_seed(42)
x = torch.randn(B, H, N_RES, N_RES, device=device, dtype=torch.float32).contiguous()

native_softmax = nn.Softmax(dim=-1).to(device)

for _ in range(20):
    _ = native_softmax(x)
torch.cuda.synchronize()

for _ in range(100):
    out = native_softmax(x)
torch.cuda.synchronize()