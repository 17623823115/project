import torch
from cuda_ops import CudaSoftmax

B, H, N_RES = 1, 12, 512
device = torch.device("cuda")

torch.manual_seed(42)
x = torch.randn(B, H, N_RES, N_RES, device=device, dtype=torch.float32).contiguous()

CudaSoftmax.use_cuda = True
cuda_softmax = CudaSoftmax(dim=-1).to(device)

for _ in range(20):
    _ = cuda_softmax(x)
torch.cuda.synchronize()

for _ in range(100):
    out = cuda_softmax(x)
torch.cuda.synchronize()