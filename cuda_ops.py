import os
import torch
import cuda_kernels

# ===================== LayerNorm =====================
def cuda_layernorm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    C = orig_shape[-1]
    M = x.numel() // C
    x_flat = x.reshape(M, C).contiguous()
    
    if C == 128:
        out_flat = cuda_kernels.cuda_layernorm_128(x_flat, weight, bias, eps)
    elif C == 384:
        out_flat = cuda_kernels.cuda_layernorm_384(x_flat, weight, bias, eps)
    else:
        out_flat = cuda_kernels.cuda_layernorm_generic(x_flat, weight, bias, eps)
    
    return out_flat.reshape(orig_shape)

# ===================== Softmax =====================
class CudaSoftmax(torch.nn.Module):
    use_cuda = False
    
    def __init__(self, dim=-1):
        super().__init__()
        assert dim == -1, "仅支持 dim=-1"
        self.dim = dim
    
    def forward(self, x, mask=None):
        if not self.use_cuda:
            if mask is not None:
                x = x + mask
            return torch.nn.functional.softmax(x, dim=self.dim)
        
        x = x.contiguous()
        if mask is not None:
            mask = mask.contiguous()
            return cuda_kernels.cuda_softmax_masked(x, mask)
        else:
            return cuda_kernels.cuda_softmax(x)

# =====================  GEMM =====================
def cuda_linear_gemm(x, weight_T, bias, relu=False, allow_tf32=True):
    """
    x: [M, K]
    weight_T: [K, N] 
    bias: [N]
    return: [M, N]
    """
    return cuda_kernels.cuda_gemm_bias_relu(x, weight_T, bias, relu)

def cuda_linear_gemm_residual(x, weight_T, bias, residual, relu=False, allow_tf32=True):
    return cuda_kernels.cuda_gemm_bias_residual(x, weight_T, bias, residual, relu)

# ===================== Fusion operator =====================
def cuda_fused_residual_ln_linear_relu(s, ipa_out, ln_weight, ln_bias, lin_weight, lin_bias, eps=1e-5, allow_tf32=True):
    """
    Four-in-one integration: Residual addition + LayerNorm + Linear + ReLU
    return (ln_out, lin_out)
    """
    ln_out, lin_out = cuda_kernels.cuda_fused_residual_ln_linear(
        s, ipa_out, ln_weight, ln_bias, lin_weight, lin_bias, eps
    )
    return ln_out, lin_out

def cuda_fused_two_linear_residual(x, w2, b2, w3, b3, residual, allow_tf32=True):
    """
    Two-layer linear fusion: Linear2 + ReLU + Linear3 + Residual
    """
    return cuda_kernels.cuda_fused_two_linear_residual(x, w2, b2, w3, b3, residual)

def cuda_point_sq_dist(q_pts, k_pts):
    return cuda_kernels.cuda_point_sq_dist(q_pts, k_pts)

def cuda_fused_attention_softmax(q, k, extra_att, mask, scalar_scale):
    return cuda_kernels.cuda_fused_attention_softmax(q, k, extra_att, mask, scalar_scale)

def cuda_fused_linear_point_transform(x, weight_T, bias, rot_mats, trans, HP, allow_tf32=True):
    return cuda_kernels.cuda_fused_linear_point_transform(x, weight_T, bias, rot_mats, trans, HP)