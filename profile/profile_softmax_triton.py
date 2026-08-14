import torch
from triton_softmax import Softmax

B, H, N_RES = 1, 12, 512
device = torch.device("cuda")

torch.manual_seed(42)
x = torch.randn(B, H, N_RES, N_RES, device=device, dtype=torch.float32).contiguous()

Softmax.use_triton = True
triton_softmax = Softmax(dim=-1).to(device)

for _ in range(20):
    _ = triton_softmax(x)
torch.cuda.synchronize()

for _ in range(100):
    out = triton_softmax(x)
torch.cuda.synchronize()