import math
import torch
import torch.nn as nn
from triton_kernels import (
    triton_linear_gemm,
    triton_fused_attention_softmax,
    triton_linear_gemm_residual,
    triton_fused_residual_ln_linear_relu,
    triton_fused_two_linear_residual,
    triton_fused_linear_point_transform,
    triton_point_sq_dist
)
from triton_softmax import Softmax
from layernorm__triton import triton_layernorm
from cuda_ops import (
    cuda_linear_gemm,
    cuda_linear_gemm_residual,
    cuda_fused_attention_softmax,
    cuda_fused_residual_ln_linear_relu,
    cuda_fused_two_linear_residual,
    cuda_fused_linear_point_transform,
    cuda_point_sq_dist,
    CudaSoftmax,
    cuda_layernorm
)

# ===================== 基础算子（替换靶点完整保留）=====================
# class Linear(nn.Linear):
#     # 全局开关
#     use_triton = False
#     use_tf32 = True  # TF32 开关：True=启用，False=严格FP32

#     def __init__(self, in_features, out_features, bias=True, init="default", precision=None):
#         super().__init__(in_features, out_features, bias=bias)
#         self._init_type = init
        
#         if init == "relu":
#             nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
#             if self.bias is not None:
#                 nn.init.zeros_(self.bias)
#         elif init == "final":
#             nn.init.zeros_(self.weight)
#             if self.bias is not None:
#                 nn.init.zeros_(self.bias)
#         # 提前转置并缓存权重
#         self._cached_weight_T = None

#     def forward(self, x):
#         if not Linear.use_triton:
#             # 原生 PyTorch 路径
#             return super().forward(x)
#         else:
#             if self._cached_weight_T is None or self._cached_weight_T.device != x.device:
#                 self._cached_weight_T = self.weight.T.contiguous()
            
#             orig_shape = x.shape
#             M = x.numel() // orig_shape[-1]
#             x_flat = x.reshape(M, self.in_features)
            
#             fuse_relu = self._init_type == "relu"
#             # 传递 TF32 开关到内核
#             out_flat = triton_linear_gemm(
#                 x_flat, self._cached_weight_T, self.bias,
#                 relu=fuse_relu, allow_tf32=Linear.use_tf32
#             )
#             return out_flat.reshape(*orig_shape[:-1], self.out_features)
        
#     def forward_residual(self, x, residual):
#         """残差融合版：output = (x @ W + bias).relu() + residual
#         仅在 init="final" 的层使用，对应残差块的最后一个 Linear
#         """
#         if not Linear.use_triton:
#             # 原生路径兜底
#             out = super().forward(x)
#             if self._init_type == "relu":
#                 out = torch.relu(out)
#             return out + residual
#         else:
#             if self._cached_weight_T is None or self._cached_weight_T.device != x.device:
#                 self._cached_weight_T = self.weight.T.contiguous()
            
#             from triton_kernels import triton_linear_gemm_residual
#             orig_shape = x.shape
#             M = x.numel() // orig_shape[-1]
#             x_flat = x.reshape(M, self.in_features)
#             res_flat = residual.reshape(M, self.out_features)
            
#             fuse_relu = self._init_type == "relu"
#             out_flat = triton_linear_gemm_residual(
#                 x_flat, self._cached_weight_T, self.bias, res_flat,
#                 relu=fuse_relu, allow_tf32=Linear.use_tf32
#             )
#             return out_flat.reshape(*orig_shape[:-1], self.out_features)
class Linear(nn.Linear):
    # Global switch: Select from three options. CUDA priority > Triton > Native
    use_triton = False
    use_cuda = False
    use_tf32 = True

    def __init__(self, in_features, out_features, bias=True, init="default", precision=None):
        super().__init__(in_features, out_features, bias=bias)
        self._init_type = init
        if init == "relu":
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif init == "final":
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        self._cached_weight_T = None

    def forward(self, x):
        # CUDA 
        if self.use_cuda:
            if self._cached_weight_T is None or self._cached_weight_T.device != x.device:
                self._cached_weight_T = self.weight.T.contiguous()
            orig_shape = x.shape
            M = x.numel() // orig_shape[-1]
            x_flat = x.reshape(M, self.in_features)
            fuse_relu = self._init_type == "relu"
            out_flat = cuda_linear_gemm(
                x_flat, self._cached_weight_T, self.bias,
                relu=fuse_relu, allow_tf32=self.use_tf32
            )
            return out_flat.reshape(*orig_shape[:-1], self.out_features)
        # Triton 
        elif self.use_triton:
            if self._cached_weight_T is None or self._cached_weight_T.device != x.device:
                self._cached_weight_T = self.weight.T.contiguous()
            orig_shape = x.shape
            M = x.numel() // orig_shape[-1]
            x_flat = x.reshape(M, self.in_features)
            fuse_relu = self._init_type == "relu"
            out_flat = triton_linear_gemm(
                x_flat, self._cached_weight_T, self.bias,
                relu=fuse_relu, allow_tf32=self.use_tf32
            )
            return out_flat.reshape(*orig_shape[:-1], self.out_features)
        # native PyTorch
        else:
            return super().forward(x)

    def forward_residual(self, x, residual):
        if self.use_cuda:
            if self._cached_weight_T is None or self._cached_weight_T.device != x.device:
                self._cached_weight_T = self.weight.T.contiguous()
            orig_shape = x.shape
            M = x.numel() // orig_shape[-1]
            x_flat = x.reshape(M, self.in_features)
            res_flat = residual.reshape(M, self.out_features)
            fuse_relu = self._init_type == "relu"
            out_flat = cuda_linear_gemm_residual(
                x_flat, self._cached_weight_T, self.bias, res_flat,
                relu=fuse_relu, allow_tf32=self.use_tf32
            )
            return out_flat.reshape(*orig_shape[:-1], self.out_features)
        elif self.use_triton:
            if self._cached_weight_T is None or self._cached_weight_T.device != x.device:
                self._cached_weight_T = self.weight.T.contiguous()
            orig_shape = x.shape
            M = x.numel() // orig_shape[-1]
            x_flat = x.reshape(M, self.in_features)
            res_flat = residual.reshape(M, self.out_features)
            fuse_relu = self._init_type == "relu"
            out_flat = triton_linear_gemm_residual(
                x_flat, self._cached_weight_T, self.bias, res_flat,
                relu=fuse_relu, allow_tf32=self.use_tf32
            )
            return out_flat.reshape(*orig_shape[:-1], self.out_features)
        else:
            out = super().forward(x)
            if self._init_type == "relu":
                out = torch.relu(out)
            return out + residual

# class LayerNorm(nn.LayerNorm):
#     # 全局开关：False=用原生PyTorch，True=用Triton实现
#     use_triton = False
#     def __init__(self, normalized_shape, eps=1e-5):
#         super().__init__(normalized_shape, eps=eps)
    
#     def forward(self, x):
#         if not LayerNorm.use_triton:
#             # 原生 PyTorch 路径
#             return super().forward(x)
#         else:
#             # Triton 路径：复用原生权重与偏置
#             return triton_layernorm(x, self.weight, self.bias, self.eps)

class LayerNorm(nn.LayerNorm):
    use_triton = False
    use_cuda = False

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x):
        if self.use_cuda:
            return cuda_layernorm(x, self.weight, self.bias, self.eps)
        elif self.use_triton:
            return triton_layernorm(x, self.weight, self.bias, self.eps)
        else:
            return super().forward(x)
        
def ipa_point_weights_init_(tensor):
    nn.init.constant_(tensor, 0.1)

# ===================== submodule =====================
class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        super().__init__()
        self.linear_1 = Linear(c_hidden, c_hidden, init="relu")
        self.linear_2 = Linear(c_hidden, c_hidden, init="final")
        self.relu = nn.ReLU()

    def forward(self, a):
        s_initial = a
        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2.forward_residual(a, s_initial)
        return a

class AngleResnet(nn.Module):
    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        super().__init__()
        self.linear_in = Linear(c_in, c_hidden)
        self.linear_initial = Linear(c_in, c_hidden)
        self.layers = nn.ModuleList([AngleResnetBlock(c_hidden) for _ in range(no_blocks)])
        self.linear_out = Linear(c_hidden, no_angles * 2)
        self.relu = nn.ReLU()
        self.eps = epsilon

    def forward(self, s, s_initial):
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial
        for l in self.layers:
            s = l(s)
        s = self.relu(s)
        s = self.linear_out(s)
        s = s.view(s.shape[:-1] + (-1, 2))
        unnormalized_s = s
        norm_denom = torch.sqrt(torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True), min=self.eps))
        s = s / norm_denom
        return unnormalized_s, s

# class PointProjection(nn.Module):
#     def __init__(self, c_hidden, num_points, no_heads):
#         super().__init__()
#         self.no_heads = no_heads
#         self.num_points = num_points
#         self.HP = no_heads * num_points  # 每个残基总点数
#         self.linear = Linear(c_hidden, no_heads * 3 * num_points)

#     def forward(self, activations, rot_mats, trans):
#         B, N = activations.shape[:2]
#         # CUDA 融合路径
#         if Linear.use_cuda:
#             if self.linear._cached_weight_T is None or self.linear._cached_weight_T.device != activations.device:
#                 self.linear._cached_weight_T = self.linear.weight.T.contiguous()
#             x_flat = activations.reshape(B * N, -1).contiguous()
#             rot_flat = rot_mats.reshape(B * N, 3, 3).contiguous()
#             trans_flat = trans.reshape(B * N, 3).contiguous()
#             points_flat_global = cuda_fused_linear_point_transform(
#                 x_flat, self.linear._cached_weight_T, self.linear.bias,
#                 rot_flat, trans_flat, self.HP,
#                 allow_tf32=Linear.use_tf32
#             )
#             return points_flat_global.reshape(B, N, self.no_heads, self.num_points, 3)
#         # Triton 融合路径
#         elif Linear.use_triton:
#             if self.linear._cached_weight_T is None or self.linear._cached_weight_T.device != activations.device:
#                 self.linear._cached_weight_T = self.linear.weight.T.contiguous()
#             x_flat = activations.reshape(B * N, -1).contiguous()
#             rot_flat = rot_mats.reshape(B * N, 3, 3).contiguous()
#             trans_flat = trans.reshape(B * N, 3).contiguous()
#             points_flat_global = triton_fused_linear_point_transform(
#                 x_flat, self.linear._cached_weight_T, self.linear.bias,
#                 rot_flat, trans_flat, self.HP,
#                 allow_tf32=Linear.use_tf32
#             )
#             return points_flat_global.reshape(B, N, self.no_heads, self.num_points, 3)
#         # 原生路径
#         else:
#             points_local = self.linear(activations).view(B, N, self.no_heads, self.num_points, 3)
#             points_flat = points_local.reshape(B, N, -1, 3)
#             points_transformed = points_flat @ rot_mats.mT + trans.unsqueeze(-2)
#             points_global = points_transformed.reshape(B, N, self.no_heads, self.num_points, 3)
#             return points_global

class PointProjection(nn.Module):
    def __init__(self, c_hidden, num_points, no_heads):
        super().__init__()
        self.no_heads = no_heads
        self.num_points = num_points
        self.linear = Linear(c_hidden, no_heads * 3 * num_points)

    def forward(self, activations, rot_mats, trans):
        B, N = activations.shape[:2]
        # Generate local point coordinates, shape: [B, N, H, P, 3] (5 dimensions, fixed dimension count)
        points_local = self.linear(activations).view(B, N, self.no_heads, self.num_points, 3)
        
        # Merge the head and point dimensions, perform rigid body transformation uniformly, 
        # and avoid errors caused by high-dimensional broadcasting.
        points_flat = points_local.reshape(B, N, -1, 3)  # [B, N, H*P, 3]
        # Rigid body transformation: Rotation + Translation
        points_transformed = points_flat @ rot_mats.mT + trans.unsqueeze(-2)
        # Restore the head and point dimensions, output fixed 5D: [B, N, H, P, 3]
        points_global = points_transformed.reshape(B, N, self.no_heads, self.num_points, 3)
        return points_global

# class InvariantPointAttention(nn.Module):
#     use_fused_attention = False
    
#     def __init__(self, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points, inf=1e5, eps=1e-8):
#         super().__init__()
#         self.c_s = c_s
#         self.c_z = c_z
#         self.c_hidden = c_hidden
#         self.no_heads = no_heads
#         self.no_qk_points = no_qk_points
#         self.no_v_points = no_v_points
#         self.inf = inf
#         self.eps = eps
#         hc = c_hidden * no_heads
#         # GEMM 替换靶点
#         self.linear_q = Linear(c_s, hc)
#         self.linear_kv = Linear(c_s, 2 * hc)
#         self.linear_q_points = PointProjection(c_s, no_qk_points, no_heads)
#         self.linear_kv_points = PointProjection(c_s, no_qk_points + no_v_points, no_heads)
#         self.linear_b = Linear(c_z, no_heads)
#         concat_out_dim = no_heads * (c_z + c_hidden + no_v_points * 4)
#         self.linear_out = Linear(concat_out_dim, c_s, init="final")
#         self.head_weights = nn.Parameter(torch.zeros((no_heads)))
#         ipa_point_weights_init_(self.head_weights)
#         self.softmax = Softmax(dim=-1) # Softmax / FlashAttention 替换靶点
#         self.softplus = nn.Softplus()

#     def forward(self, s, z, rot_mats, trans, mask):
#         B, N_res, _ = s.shape
#         # ========== 1. 标量 Q/K/V ==========
#         q = self.linear_q(s).view(B, N_res, self.no_heads, self.c_hidden)
#         kv = self.linear_kv(s).view(B, N_res, self.no_heads, 2 * self.c_hidden)
#         k, v = torch.split(kv, self.c_hidden, dim=-1)
#         # ========== 2. 点 Q/K/V ==========
#         q_pts = self.linear_q_points(s, rot_mats, trans)
#         kv_pts = self.linear_kv_points(s, rot_mats, trans)
#         k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

#         # ========== 融合路径（开关开启时执行）==========
#         if InvariantPointAttention.use_fused_attention and Softmax.use_triton:
#             # 1. 预计算所有加性偏置（配对特征偏置 + 点注意力分数）
#             b = self.linear_b(z) * math.sqrt(1.0 / 3)
#             b = b.permute(0, 3, 1, 2)  # [B, H, N, N]

#             # 展平头维度，调用融合内核
#             q_pts_flat = q_pts.permute(0, 2, 1, 3, 4).reshape(B * self.no_heads, N_res, self.no_qk_points, 3).contiguous()
#             k_pts_flat = k_pts.permute(0, 2, 1, 3, 4).reshape(B * self.no_heads, N_res, self.no_qk_points, 3).contiguous()
#             pt_att_flat = triton_point_sq_dist(q_pts_flat, k_pts_flat)
#             pt_att = pt_att_flat.reshape(B, self.no_heads, N_res, N_res)

#             head_weights = self.softplus(self.head_weights).view(1, self.no_heads, 1, 1)
#             pt_scale = math.sqrt(1.0 / (3 * (self.no_qk_points * 9 / 2)))
#             pt_att = pt_att * head_weights * pt_scale * (-0.5)

#             # 合并所有加性偏置，展平为 [B*H, N, N]
#             extra_att = (b + pt_att).reshape(B * self.no_heads, N_res, N_res).contiguous()

#             # 2. Q/K 展平为 [B*H, N, C]
#             q_flat = q.permute(0, 2, 1, 3).reshape(B * self.no_heads, N_res, self.c_hidden).contiguous()
#             k_flat = k.permute(0, 2, 1, 3).reshape(B * self.no_heads, N_res, self.c_hidden).contiguous()

#             # 3. 生成掩码：扩展到头维度，展平为 [B*H, N, N]
#             square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
#             square_mask = self.inf * (square_mask - 1)
#             # 关键修复：扩展头维度，和注意力矩阵形状严格对齐
#             square_mask = square_mask.unsqueeze(1).expand(B, self.no_heads, N_res, N_res).reshape(B * self.no_heads, N_res, N_res).contiguous()

#             # 4. 调用融合内核
#             scalar_scale = math.sqrt(1.0 / (3 * self.c_hidden))
#             a_flat = triton_fused_attention_softmax(
#                 q_flat, k_flat, extra_att, square_mask, scalar_scale
#             )
#             a = a_flat.reshape(B, self.no_heads, N_res, N_res)

#         # ========== 原生路径（开关关闭时执行）==========
#         else:
#             a = torch.einsum("b i h c, b j h c -> b h i j", q, k)
#             a *= math.sqrt(1.0 / (3 * self.c_hidden))
#             b = self.linear_b(z)
#             a += math.sqrt(1.0 / 3) * b.permute(0, 3, 1, 2)

#             q_sq = (q_pts ** 2).sum(dim=(-1, -2))
#             k_sq = (k_pts ** 2).sum(dim=(-1, -2))
#             qk = torch.einsum("b i h p c, b j h p c -> b h i j", q_pts, k_pts)
#             q_sq = q_sq.permute(0, 2, 1).unsqueeze(-1)
#             k_sq = k_sq.permute(0, 2, 1).unsqueeze(-2)
#             pt_att = q_sq + k_sq - 2 * qk

#             head_weights = self.softplus(self.head_weights).view(1, self.no_heads, 1, 1)
#             scale = math.sqrt(1.0 / (3 * (self.no_qk_points * 9 / 2)))
#             pt_att = pt_att * head_weights * scale * (-0.5)
#             a += pt_att

#             square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
#             square_mask = self.inf * (square_mask - 1)
#             #a = a + square_mask.unsqueeze(1)
#             #a = self.softmax(a)
#             a = self.softmax(a, mask=square_mask.unsqueeze(1))

#         # ========== 注意力加权求和（两版通用） ==========
#         o = torch.einsum("b h i j, b j h c -> b i h c", a, v)
#         o = o.flatten(start_dim=-2)

#         o_pt = torch.einsum("b h i j, b j h p c -> b i h p c", a, v_pts)
#         o_pt_flat = o_pt.reshape(B, N_res, -1, 3)
#         o_pt_transformed = (o_pt_flat - trans.unsqueeze(-2)) @ rot_mats
#         o_pt = o_pt_transformed.reshape(B, N_res, self.no_heads, self.no_v_points, 3)
        
#         o_pt_norm = torch.sqrt(o_pt.pow(2).sum(dim=-1) + self.eps).flatten(start_dim=-2)
#         o_pt = o_pt.flatten(start_dim=-3, end_dim=-2)
#         o_pt_list = [o_pt[..., 0], o_pt[..., 1], o_pt[..., 2]]

#         o_pair = torch.einsum("b h i j, b i j c -> b i h c", a, z)
#         o_pair = o_pair.flatten(start_dim=-2)

#         return self.linear_out(torch.cat([o, *o_pt_list, o_pt_norm, o_pair], dim=-1))

class InvariantPointAttention(nn.Module):
    use_fused_attention = False
    
    def __init__(self, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points, inf=1e5, eps=1e-8):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        hc = c_hidden * no_heads

        # GEMM replace the target point
        self.linear_q = Linear(c_s, hc)
        self.linear_kv = Linear(c_s, 2 * hc)
        self.linear_q_points = PointProjection(c_s, no_qk_points, no_heads)
        self.linear_kv_points = PointProjection(c_s, no_qk_points + no_v_points, no_heads)
        self.linear_b = Linear(c_z, no_heads)

        concat_out_dim = no_heads * (c_z + c_hidden + no_v_points * 4)
        self.linear_out = Linear(concat_out_dim, c_s, init="final")

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)
        self.softmax = Softmax(dim=-1)  # Softmax / FlashAttention replace the target point
        self.softplus = nn.Softplus()

    def forward(self, s, z, rot_mats, trans, mask):
        B, N_res, _ = s.shape

        # ========== 1. Scalar Q/K/V ==========
        q = self.linear_q(s).view(B, N_res, self.no_heads, self.c_hidden)
        kv = self.linear_kv(s).view(B, N_res, self.no_heads, 2 * self.c_hidden)
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # ========== 2. point Q/K/V ==========
        q_pts = self.linear_q_points(s, rot_mats, trans)
        kv_pts = self.linear_kv_points(s, rot_mats, trans)
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        # ========== Integrated path (CUDA / Triton dual-backend adaptive) ==========
        if InvariantPointAttention.use_fused_attention:
            # 1. Precomputed pairing feature bias b (common to both backends)
            b = self.linear_b(z) * math.sqrt(1.0 / 3)
            b = b.permute(0, 3, 1, 2)  # [B, H, N, N]

            # 2. Flatten head dimension, prepare point distance calculation (common to both backends)
            q_pts_flat = q_pts.permute(0, 2, 1, 3, 4).reshape(B * self.no_heads, N_res, self.no_qk_points, 3).contiguous()
            k_pts_flat = k_pts.permute(0, 2, 1, 3, 4).reshape(B * self.no_heads, N_res, self.no_qk_points, 3).contiguous()

            # 3. Call the fused kernel for point square distances corresponding to each backend
            if Linear.use_cuda:
                B_H, N_q, P, _ = q_pts_flat.shape
                _, N_k, _, _ = k_pts_flat.shape
                pt_att_list = []
                # Call the single-head CUDA kernel sequentially for each head
                for h in range(B_H):
                    q_h = q_pts_flat[h].contiguous()  # [N_q, P, 3]
                    k_h = k_pts_flat[h].contiguous()  # [N_k, P, 3]
                    dist_h = cuda_point_sq_dist(q_h, k_h)
                    pt_att_list.append(dist_h.unsqueeze(0))
                pt_att_flat = torch.cat(pt_att_list, dim=0)  # [B_H, N_q, N_k]
            else:
                pt_att_flat = triton_point_sq_dist(q_pts_flat, k_pts_flat)
            
            pt_att = pt_att_flat.reshape(B, self.no_heads, N_res, N_res)

            # 4. Attention weight scaling (common to both backends)
            head_weights = self.softplus(self.head_weights).view(1, self.no_heads, 1, 1)
            pt_scale = math.sqrt(1.0 / (3 * (self.no_qk_points * 9 / 2)))
            pt_att = pt_att * head_weights * pt_scale * (-0.5)

            # 5. Merge all additive biases, flatten head dimension (common to both backends)
            extra_att = (b + pt_att).reshape(B * self.no_heads, N_res, N_res).contiguous()

            # 6. Flatten Q/K dimensions, prepare attention dot product (common to both backends)
            q_flat = q.permute(0, 2, 1, 3).reshape(B * self.no_heads, N_res, self.c_hidden).contiguous()
            k_flat = k.permute(0, 2, 1, 3).reshape(B * self.no_heads, N_res, self.c_hidden).contiguous()

            # 7. Generate attention mask (common to both backends)
            square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            square_mask = self.inf * (square_mask - 1)
            # Expand to head dimension, and align with the shape of the attention matrix
            square_mask = square_mask.unsqueeze(1).expand(B, self.no_heads, N_res, N_res).reshape(B * self.no_heads, N_res, N_res).contiguous()

            # 8. Call the fused kernel for QK dot product + mask + Softmax corresponding to each backend
            scalar_scale = math.sqrt(1.0 / (3 * self.c_hidden))
            if Linear.use_cuda:
                B_H, N_q, C = q_flat.shape
                _, N_k, _ = k_flat.shape
                attn_list = []
                for h in range(B_H):
                    q_h = q_flat[h].contiguous()
                    k_h = k_flat[h].contiguous()
                    extra_h = extra_att[h].contiguous()
                    mask_h = square_mask[h].contiguous()
                    a_h = cuda_fused_attention_softmax(q_h, k_h, extra_h, mask_h, scalar_scale)
                    attn_list.append(a_h.unsqueeze(0))
                a_flat = torch.cat(attn_list, dim=0)  # [B_H, N_q, N_k]
            else:
                a_flat = triton_fused_attention_softmax(q_flat, k_flat, extra_att, square_mask, scalar_scale)

            a = a_flat.reshape(B, self.no_heads, N_res, N_res)

        # ========== Native path ==========
        else:
            a = torch.einsum("b i h c, b j h c -> b h i j", q, k)
            a *= math.sqrt(1.0 / (3 * self.c_hidden))
            b = self.linear_b(z)
            a += math.sqrt(1.0 / 3) * b.permute(0, 3, 1, 2)

            q_sq = (q_pts ** 2).sum(dim=(-1, -2))
            k_sq = (k_pts ** 2).sum(dim=(-1, -2))
            qk = torch.einsum("b i h p c, b j h p c -> b h i j", q_pts, k_pts)
            q_sq = q_sq.permute(0, 2, 1).unsqueeze(-1)
            k_sq = k_sq.permute(0, 2, 1).unsqueeze(-2)
            pt_att = q_sq + k_sq - 2 * qk

            head_weights = self.softplus(self.head_weights).view(1, self.no_heads, 1, 1)
            scale = math.sqrt(1.0 / (3 * (self.no_qk_points * 9 / 2)))
            pt_att = pt_att * head_weights * scale * (-0.5)
            a += pt_att

            square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            square_mask = self.inf * (square_mask - 1)
            a = self.softmax(a, mask=square_mask.unsqueeze(1))

        # ========== Attention weighted sum (common to both versions, fully preserving original logic) ==========
        o = torch.einsum("b h i j, b j h c -> b i h c", a, v)
        o = o.flatten(start_dim=-2)

        o_pt = torch.einsum("b h i j, b j h p c -> b i h p c", a, v_pts)
        o_pt_flat = o_pt.reshape(B, N_res, -1, 3)
        o_pt_transformed = (o_pt_flat - trans.unsqueeze(-2)) @ rot_mats
        o_pt = o_pt_transformed.reshape(B, N_res, self.no_heads, self.no_v_points, 3)
        
        o_pt_norm = torch.sqrt(o_pt.pow(2).sum(dim=-1) + self.eps).flatten(start_dim=-2)
        o_pt = o_pt.flatten(start_dim=-3, end_dim=-2)
        o_pt_list = [o_pt[..., 0], o_pt[..., 1], o_pt[..., 2]]

        o_pair = torch.einsum("b h i j, b i j c -> b i h c", a, z)
        o_pair = o_pair.flatten(start_dim=-2)

        return self.linear_out(torch.cat([o, *o_pt_list, o_pt_norm, o_pair], dim=-1))

class BackboneUpdate(nn.Module):
    def __init__(self, c_s):
        super().__init__()
        self.linear = Linear(c_s, 6, init="final")

    def forward(self, s):
        return self.linear(s)

class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.linear_1 = Linear(c, c, init="relu")
        self.linear_2 = Linear(c, c, init="relu")
        self.linear_3 = Linear(c, c, init="final")
        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3.forward_residual(s, s_initial)
        return s

class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([StructureModuleTransitionLayer(c) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(c)  # LayerNorm

    def forward(self, s):
        for l in self.layers:
            s = l(s)
        s = self.dropout(s)
        s = self.layer_norm(s)
        return s

class StructureModule(nn.Module):
    def __init__(self, c_s, c_z, c_ipa, c_resnet, no_heads_ipa,
                 no_qk_points, no_v_points, dropout_rate, no_blocks,
                 no_transition_layers, no_resnet_blocks, no_angles,
                 trans_scale_factor, epsilon, inf, is_multimer=False, **kwargs):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.no_blocks = no_blocks
        self.trans_scale_factor = trans_scale_factor

        self.layer_norm_s = LayerNorm(c_s)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_in = Linear(c_s, c_s)

        self.ipa = InvariantPointAttention(
            c_s, c_z, c_ipa, no_heads_ipa, no_qk_points, no_v_points, inf, epsilon
        )
        self.ipa_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_ipa = LayerNorm(c_s)

        self.transition = StructureModuleTransition(c_s, no_transition_layers, dropout_rate)
        self.bb_update = BackboneUpdate(c_s)
        self.angle_resnet = AngleResnet(c_s, c_resnet, no_resnet_blocks, no_angles, epsilon)

    def forward(self, evoformer_output_dict, aatype, mask=None,
                inplace_safe=False, _offload_inference=False, _z_reference_list=None):
        """
        Supports the original version's call signature. The additional parameters are only for compatibility handling.
        """
        s = evoformer_output_dict["single"]
        z = evoformer_output_dict["pair"]
        B, N_res, _ = s.shape

        if mask is None:
            mask = s.new_ones(B, N_res)

        # Input normalization
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        s_initial = s
        s = self.linear_in(s)

        # Initialize unit rigid body
        rot_mats = torch.eye(3, device=s.device, dtype=s.dtype).expand(B, N_res, 3, 3).contiguous()
        trans = torch.zeros(B, N_res, 3, device=s.device, dtype=s.dtype)

        outputs = []
        for i in range(self.no_blocks):
            ipa_out = self.ipa(s, z, rot_mats, trans, mask)
            
            # ---------- CUDA Integration path ----------
            if Linear.use_cuda and LayerNorm.use_cuda:
                B, N_res, _ = s.shape
                s_flat = s.reshape(B * N_res, -1)
                ipa_flat = ipa_out.reshape(B * N_res, -1)

                lin1 = self.transition.layers[0].linear_1
                if lin1._cached_weight_T is None or lin1._cached_weight_T.device != s.device:
                    lin1._cached_weight_T = lin1.weight.T.contiguous()

                ln_out_flat, lin1_out_flat = cuda_fused_residual_ln_linear_relu(
                    s_flat, ipa_flat,
                    self.layer_norm_ipa.weight, self.layer_norm_ipa.bias,
                    lin1._cached_weight_T, lin1.bias,
                    eps=self.layer_norm_ipa.eps,
                    allow_tf32=Linear.use_tf32
                )

                trans_residual = ln_out_flat.reshape(B, N_res, -1)
                s = lin1_out_flat.reshape(B, N_res, -1)

                trans_layer = self.transition.layers[0]
                lin2 = trans_layer.linear_2
                if lin2._cached_weight_T is None or lin2._cached_weight_T.device != s.device:
                    lin2._cached_weight_T = lin2.weight.T.contiguous()
                lin3 = trans_layer.linear_3
                if lin3._cached_weight_T is None or lin3._cached_weight_T.device != s.device:
                    lin3._cached_weight_T = lin3.weight.T.contiguous()

                B, N_res, _ = s.shape
                s_flat = s.reshape(B * N_res, -1)
                res_flat = trans_residual.reshape(B * N_res, -1)

                s_flat = cuda_fused_two_linear_residual(
                    s_flat,
                    lin2._cached_weight_T, lin2.bias,
                    lin3._cached_weight_T, lin3.bias,
                    res_flat,
                    allow_tf32=Linear.use_tf32
                )
                s = s_flat.reshape(B, N_res, -1)
                s = self.transition.dropout(s)
                s = self.transition.layer_norm(s)
            # ---------- Integration path: Both Linear and LayerNorm become effective when Triton is enabled. ----------
            if Linear.use_triton and LayerNorm.use_triton:
                B, N_res, _ = s.shape
                s_flat = s.reshape(B * N_res, -1)
                ipa_flat = ipa_out.reshape(B * N_res, -1)
                
                # Reuse the transposed weight cache of the Linear class
                lin1 = self.transition.layers[0].linear_1
                if lin1._cached_weight_T is None or lin1._cached_weight_T.device != s.device:
                    lin1._cached_weight_T = lin1.weight.T.contiguous()
                
                # Invoke the fusion kernel, simultaneously obtaining the output of LayerNorm 
                # (the residual source of the Transition layer) and the output of Linear1 + ReLU
                ln_out_flat, lin1_out_flat = triton_fused_residual_ln_linear_relu(
                    s_flat, ipa_flat,
                    self.layer_norm_ipa.weight, self.layer_norm_ipa.bias,
                    lin1._cached_weight_T, lin1.bias,
                    eps=self.layer_norm_ipa.eps,
                    allow_tf32=Linear.use_tf32
                )
                
                # Restore shape, trans_residual refers to the residual source of the Transition layer, 
                # and does not have the same name as the outer global s_initial.
                trans_residual = ln_out_flat.reshape(B, N_res, -1)
                s = lin1_out_flat.reshape(B, N_res, -1)
                
                # Manually perform the remaining calculation of 
                # Transition: Linear2 + ReLU + Linear3 + Residual, using a dual-layer fusion kernel
                trans_layer = self.transition.layers[0]

                # Pre-cache the transposed weights of two Linear layers
                lin2 = trans_layer.linear_2
                if lin2._cached_weight_T is None or lin2._cached_weight_T.device != s.device:
                    lin2._cached_weight_T = lin2.weight.T.contiguous()
                lin3 = trans_layer.linear_3
                if lin3._cached_weight_T is None or lin3._cached_weight_T.device != s.device:
                    lin3._cached_weight_T = lin3.weight.T.contiguous()

                B, N_res, _ = s.shape
                s_flat = s.reshape(B * N_res, -1)
                res_flat = trans_residual.reshape(B * N_res, -1)

                # Call the dual-layer fusion kernel: Linear2 + ReLU + Linear3 + Residual
                s_flat = triton_fused_two_linear_residual(
                    s_flat,
                    lin2._cached_weight_T, lin2.bias,
                    lin3._cached_weight_T, lin3.bias,
                    res_flat,
                    allow_tf32=Linear.use_tf32
                )
                s = s_flat.reshape(B, N_res, -1)

                s = self.transition.dropout(s)
                s = self.transition.layer_norm(s)
            # ---------- Native path: Rollback when the switch is not fully turned on ----------
            else:
                s = s + ipa_out
                s = self.ipa_dropout(s)
                s = self.layer_norm_ipa(s)
                s = self.transition(s)

            # Update the rigid body (simplified: only update the translation, ensuring the process is complete)
            update = self.bb_update(s)
            trans = trans + update[..., :3] * self.trans_scale_factor

            # Angle prediction + coordinate output
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)
            pred_xyz = torch.randn(B, N_res, 14, 3, device=s.device, dtype=s.dtype)
            sidechain_frames = torch.eye(4, device=s.device, dtype=s.dtype).expand(B, N_res, 8, 4, 4)

            preds = {
                "frames": torch.cat([rot_mats.flatten(-2), trans], dim=-1),
                "sidechain_frames": sidechain_frames,
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }
            outputs.append(preds)

        # Stack the outputs of all the blocks
        outputs = {k: torch.stack([d[k] for d in outputs], dim=0) for k in outputs[0].keys()}
        outputs["single"] = s
        return outputs

# import math
# import torch
# import torch.nn as nn
# from triton_kernels import (
#     triton_linear_gemm,
#     triton_fused_attention_softmax,
#     triton_fused_residual_ln_linear_relu,
#     triton_fused_two_linear_residual,
#     triton_fused_linear_point_transform,
#     triton_point_sq_dist
# )
# from triton_softmax import Softmax
# from layernorm__triton import triton_layernorm

# # ===================== 基础算子（替换靶点完整保留）=====================
# class Linear(nn.Linear):
#     # 全局开关
#     use_triton = False
#     use_tf32 = True  # TF32 开关：True=启用，False=严格FP32

#     def __init__(self, in_features, out_features, bias=True, init="default", precision=None):
#         super().__init__(in_features, out_features, bias=bias)
#         self._init_type = init
        
#         if init == "relu":
#             nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
#             if self.bias is not None:
#                 nn.init.zeros_(self.bias)
#         elif init == "final":
#             nn.init.zeros_(self.weight)
#             if self.bias is not None:
#                 nn.init.zeros_(self.bias)
#         # 提前转置并缓存权重
#         self._cached_weight_T = None

#     def forward(self, x):
#         if not Linear.use_triton:
#             # 原生 PyTorch 路径
#             return super().forward(x)
#         else:
#             if self._cached_weight_T is None or self._cached_weight_T.device != x.device:
#                 self._cached_weight_T = self.weight.T.contiguous()
            
#             orig_shape = x.shape
#             M = x.numel() // orig_shape[-1]
#             x_flat = x.reshape(M, self.in_features)
            
#             fuse_relu = self._init_type == "relu"
#             # 传递 TF32 开关到内核
#             out_flat = triton_linear_gemm(
#                 x_flat, self._cached_weight_T, self.bias,
#                 relu=fuse_relu, allow_tf32=Linear.use_tf32
#             )
#             return out_flat.reshape(*orig_shape[:-1], self.out_features)
        
#     def forward_residual(self, x, residual):
#         """残差融合版：output = (x @ W + bias).relu() + residual
#         仅在 init="final" 的层使用，对应残差块的最后一个 Linear
#         """
#         if not Linear.use_triton:
#             # 原生路径兜底
#             out = super().forward(x)
#             if self._init_type == "relu":
#                 out = torch.relu(out)
#             return out + residual
#         else:
#             if self._cached_weight_T is None or self._cached_weight_T.device != x.device:
#                 self._cached_weight_T = self.weight.T.contiguous()
            
#             from triton_kernels import triton_linear_gemm_residual
#             orig_shape = x.shape
#             M = x.numel() // orig_shape[-1]
#             x_flat = x.reshape(M, self.in_features)
#             res_flat = residual.reshape(M, self.out_features)
            
#             fuse_relu = self._init_type == "relu"
#             out_flat = triton_linear_gemm_residual(
#                 x_flat, self._cached_weight_T, self.bias, res_flat,
#                 relu=fuse_relu, allow_tf32=Linear.use_tf32
#             )
#             return out_flat.reshape(*orig_shape[:-1], self.out_features)

# class LayerNorm(nn.LayerNorm):
#     # 全局开关：False=用原生PyTorch，True=用Triton实现
#     use_triton = False
#     def __init__(self, normalized_shape, eps=1e-5):
#         super().__init__(normalized_shape, eps=eps)
    
#     def forward(self, x):
#         if not LayerNorm.use_triton:
#             # 原生 PyTorch 路径
#             return super().forward(x)
#         else:
#             # Triton 路径：复用原生权重与偏置
#             return triton_layernorm(x, self.weight, self.bias, self.eps)

# def ipa_point_weights_init_(tensor):
#     nn.init.constant_(tensor, 0.1)

# # ===================== 子模块 =====================
# class AngleResnetBlock(nn.Module):
#     def __init__(self, c_hidden):
#         super().__init__()
#         self.linear_1 = Linear(c_hidden, c_hidden, init="relu")
#         self.linear_2 = Linear(c_hidden, c_hidden, init="final")
#         self.relu = nn.ReLU()

#     def forward(self, a):
#         s_initial = a
#         a = self.relu(a)
#         a = self.linear_1(a)
#         a = self.relu(a)
#         a = self.linear_2.forward_residual(a, s_initial)
#         return a

# class AngleResnet(nn.Module):
#     def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
#         super().__init__()
#         self.linear_in = Linear(c_in, c_hidden)
#         self.linear_initial = Linear(c_in, c_hidden)
#         self.layers = nn.ModuleList([AngleResnetBlock(c_hidden) for _ in range(no_blocks)])
#         self.linear_out = Linear(c_hidden, no_angles * 2)
#         self.relu = nn.ReLU()
#         self.eps = epsilon

#     def forward(self, s, s_initial):
#         s_initial = self.relu(s_initial)
#         s_initial = self.linear_initial(s_initial)
#         s = self.relu(s)
#         s = self.linear_in(s)
#         s = s + s_initial
#         for l in self.layers:
#             s = l(s)
#         s = self.relu(s)
#         s = self.linear_out(s)
#         s = s.view(s.shape[:-1] + (-1, 2))
#         unnormalized_s = s
#         norm_denom = torch.sqrt(torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True), min=self.eps))
#         s = s / norm_denom
#         return unnormalized_s, s

# class PointProjection(nn.Module):
#     def __init__(self, c_hidden, num_points, no_heads):
#         super().__init__()
#         self.no_heads = no_heads
#         self.num_points = num_points
#         self.HP = no_heads * num_points  # 每个残基总点数
#         self.linear = Linear(c_hidden, no_heads * 3 * num_points)

#     def forward(self, activations, rot_mats, trans):
#         B, N = activations.shape[:2]

#         if not Linear.use_triton:
#             # 原生 PyTorch 路径（完全兼容原逻辑）
#             points_local = self.linear(activations).view(B, N, self.no_heads, self.num_points, 3)
#             points_flat = points_local.reshape(B, N, -1, 3)
#             points_transformed = points_flat @ rot_mats.mT + trans.unsqueeze(-2)
#             points_global = points_transformed.reshape(B, N, self.no_heads, self.num_points, 3)
#             return points_global
#         else:
#             # Triton 融合路径：Linear + 刚体变换单内核完成
#             if self.linear._cached_weight_T is None or self.linear._cached_weight_T.device != activations.device:
#                 self.linear._cached_weight_T = self.linear.weight.T.contiguous()

#             # 统一展平为内核要求的维度
#             x_flat = activations.reshape(B * N, -1).contiguous()
#             rot_flat = rot_mats.reshape(B * N, 3, 3).contiguous()
#             trans_flat = trans.reshape(B * N, 3).contiguous()

#             # 调用融合内核
#             points_flat_global = triton_fused_linear_point_transform(
#                 x_flat, self.linear._cached_weight_T, self.linear.bias,
#                 rot_flat, trans_flat, self.HP,
#                 allow_tf32=Linear.use_tf32
#             )

#             # 恢复为 5 维输出格式 [B, N, H, P, 3]
#             points_global = points_flat_global.reshape(B, N, self.no_heads, self.num_points, 3)
#             return points_global

# # class PointProjection(nn.Module):
# #     def __init__(self, c_hidden, num_points, no_heads):
# #         super().__init__()
# #         self.no_heads = no_heads
# #         self.num_points = num_points
# #         self.linear = Linear(c_hidden, no_heads * 3 * num_points)

# #     def forward(self, activations, rot_mats, trans):
# #         B, N = activations.shape[:2]
# #         # 生成局部点坐标，shape: [B, N, H, P, 3]（5维，维度数固定）
# #         points_local = self.linear(activations).view(B, N, self.no_heads, self.num_points, 3)
        
# #         # 合并头和点维度，统一做刚体变换，避免高维广播出错
# #         points_flat = points_local.reshape(B, N, -1, 3)  # [B, N, H*P, 3]
# #         # 刚体变换：旋转 + 平移
# #         points_transformed = points_flat @ rot_mats.mT + trans.unsqueeze(-2)
# #         # 恢复头和点维度，输出固定5维: [B, N, H, P, 3]
# #         points_global = points_transformed.reshape(B, N, self.no_heads, self.num_points, 3)
# #         return points_global

# class InvariantPointAttention(nn.Module):
#     use_fused_attention = False
    
#     def __init__(self, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points, inf=1e5, eps=1e-8):
#         super().__init__()
#         self.c_s = c_s
#         self.c_z = c_z
#         self.c_hidden = c_hidden
#         self.no_heads = no_heads
#         self.no_qk_points = no_qk_points
#         self.no_v_points = no_v_points
#         self.inf = inf
#         self.eps = eps
#         hc = c_hidden * no_heads
#         # GEMM 替换靶点
#         self.linear_q = Linear(c_s, hc)
#         self.linear_kv = Linear(c_s, 2 * hc)
#         self.linear_q_points = PointProjection(c_s, no_qk_points, no_heads)
#         self.linear_kv_points = PointProjection(c_s, no_qk_points + no_v_points, no_heads)
#         self.linear_b = Linear(c_z, no_heads)
#         concat_out_dim = no_heads * (c_z + c_hidden + no_v_points * 4)
#         self.linear_out = Linear(concat_out_dim, c_s, init="final")
#         self.head_weights = nn.Parameter(torch.zeros((no_heads)))
#         ipa_point_weights_init_(self.head_weights)
#         self.softmax = Softmax(dim=-1) # Softmax / FlashAttention 替换靶点
#         self.softplus = nn.Softplus()

#     def forward(self, s, z, rot_mats, trans, mask):
#         B, N_res, _ = s.shape
#         # ========== 1. 标量 Q/K/V ==========
#         q = self.linear_q(s).view(B, N_res, self.no_heads, self.c_hidden)
#         kv = self.linear_kv(s).view(B, N_res, self.no_heads, 2 * self.c_hidden)
#         k, v = torch.split(kv, self.c_hidden, dim=-1)
#         # ========== 2. 点 Q/K/V ==========
#         q_pts = self.linear_q_points(s, rot_mats, trans)
#         kv_pts = self.linear_kv_points(s, rot_mats, trans)
#         k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

#         # ========== 融合路径（开关开启时执行）==========
#         if InvariantPointAttention.use_fused_attention and Softmax.use_triton:
#             # 1. 预计算所有加性偏置（配对特征偏置 + 点注意力分数）
#             b = self.linear_b(z) * math.sqrt(1.0 / 3)
#             b = b.permute(0, 3, 1, 2)  # [B, H, N, N]

#             # 展平头维度，调用融合内核
#             q_pts_flat = q_pts.permute(0, 2, 1, 3, 4).reshape(B * self.no_heads, N_res, self.no_qk_points, 3).contiguous()
#             k_pts_flat = k_pts.permute(0, 2, 1, 3, 4).reshape(B * self.no_heads, N_res, self.no_qk_points, 3).contiguous()
#             pt_att_flat = triton_point_sq_dist(q_pts_flat, k_pts_flat)
#             pt_att = pt_att_flat.reshape(B, self.no_heads, N_res, N_res)

#             head_weights = self.softplus(self.head_weights).view(1, self.no_heads, 1, 1)
#             pt_scale = math.sqrt(1.0 / (3 * (self.no_qk_points * 9 / 2)))
#             pt_att = pt_att * head_weights * pt_scale * (-0.5)

#             # 合并所有加性偏置，展平为 [B*H, N, N]
#             extra_att = (b + pt_att).reshape(B * self.no_heads, N_res, N_res).contiguous()

#             # 2. Q/K 展平为 [B*H, N, C]
#             q_flat = q.permute(0, 2, 1, 3).reshape(B * self.no_heads, N_res, self.c_hidden).contiguous()
#             k_flat = k.permute(0, 2, 1, 3).reshape(B * self.no_heads, N_res, self.c_hidden).contiguous()

#             # 3. 生成掩码：扩展到头维度，展平为 [B*H, N, N]
#             square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
#             square_mask = self.inf * (square_mask - 1)
#             # 关键修复：扩展头维度，和注意力矩阵形状严格对齐
#             square_mask = square_mask.unsqueeze(1).expand(B, self.no_heads, N_res, N_res).reshape(B * self.no_heads, N_res, N_res).contiguous()

#             # 4. 调用融合内核
#             scalar_scale = math.sqrt(1.0 / (3 * self.c_hidden))
#             a_flat = triton_fused_attention_softmax(
#                 q_flat, k_flat, extra_att, square_mask, scalar_scale
#             )
#             a = a_flat.reshape(B, self.no_heads, N_res, N_res)

#         # ========== 原生路径（开关关闭时执行）==========
#         else:
#             a = torch.einsum("b i h c, b j h c -> b h i j", q, k)
#             a *= math.sqrt(1.0 / (3 * self.c_hidden))
#             b = self.linear_b(z)
#             a += math.sqrt(1.0 / 3) * b.permute(0, 3, 1, 2)

#             q_sq = (q_pts ** 2).sum(dim=(-1, -2))
#             k_sq = (k_pts ** 2).sum(dim=(-1, -2))
#             qk = torch.einsum("b i h p c, b j h p c -> b h i j", q_pts, k_pts)
#             q_sq = q_sq.permute(0, 2, 1).unsqueeze(-1)
#             k_sq = k_sq.permute(0, 2, 1).unsqueeze(-2)
#             pt_att = q_sq + k_sq - 2 * qk

#             head_weights = self.softplus(self.head_weights).view(1, self.no_heads, 1, 1)
#             scale = math.sqrt(1.0 / (3 * (self.no_qk_points * 9 / 2)))
#             pt_att = pt_att * head_weights * scale * (-0.5)
#             a += pt_att

#             square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
#             square_mask = self.inf * (square_mask - 1)
#             #a = a + square_mask.unsqueeze(1)
#             #a = self.softmax(a)
#             a = self.softmax(a, mask=square_mask.unsqueeze(1))

#         # ========== 注意力加权求和（两版通用） ==========
#         o = torch.einsum("b h i j, b j h c -> b i h c", a, v)
#         o = o.flatten(start_dim=-2)

#         o_pt = torch.einsum("b h i j, b j h p c -> b i h p c", a, v_pts)
#         o_pt_flat = o_pt.reshape(B, N_res, -1, 3)
#         o_pt_transformed = (o_pt_flat - trans.unsqueeze(-2)) @ rot_mats
#         o_pt = o_pt_transformed.reshape(B, N_res, self.no_heads, self.no_v_points, 3)
        
#         o_pt_norm = torch.sqrt(o_pt.pow(2).sum(dim=-1) + self.eps).flatten(start_dim=-2)
#         o_pt = o_pt.flatten(start_dim=-3, end_dim=-2)
#         o_pt_list = [o_pt[..., 0], o_pt[..., 1], o_pt[..., 2]]

#         o_pair = torch.einsum("b h i j, b i j c -> b i h c", a, z)
#         o_pair = o_pair.flatten(start_dim=-2)

#         return self.linear_out(torch.cat([o, *o_pt_list, o_pt_norm, o_pair], dim=-1))

# class BackboneUpdate(nn.Module):
#     def __init__(self, c_s):
#         super().__init__()
#         self.linear = Linear(c_s, 6, init="final")

#     def forward(self, s):
#         return self.linear(s)

# class StructureModuleTransitionLayer(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.linear_1 = Linear(c, c, init="relu")
#         self.linear_2 = Linear(c, c, init="relu")
#         self.linear_3 = Linear(c, c, init="final")
#         self.relu = nn.ReLU()

#     def forward(self, s):
#         s_initial = s
#         s = self.linear_1(s)
#         s = self.relu(s)
#         s = self.linear_2(s)
#         s = self.relu(s)
#         s = self.linear_3.forward_residual(s, s_initial)
#         return s

# class StructureModuleTransition(nn.Module):
#     def __init__(self, c, num_layers, dropout_rate):
#         super().__init__()
#         self.layers = nn.ModuleList([StructureModuleTransitionLayer(c) for _ in range(num_layers)])
#         self.dropout = nn.Dropout(dropout_rate)
#         self.layer_norm = LayerNorm(c)  # LayerNorm 替换靶点

#     def forward(self, s):
#         for l in self.layers:
#             s = l(s)
#         s = self.dropout(s)
#         s = self.layer_norm(s)
#         return s

# class StructureModule(nn.Module):
#     def __init__(self, c_s, c_z, c_ipa, c_resnet, no_heads_ipa,
#                  no_qk_points, no_v_points, dropout_rate, no_blocks,
#                  no_transition_layers, no_resnet_blocks, no_angles,
#                  trans_scale_factor, epsilon, inf, is_multimer=False, **kwargs):
#         super().__init__()
#         self.c_s = c_s
#         self.c_z = c_z
#         self.no_blocks = no_blocks
#         self.trans_scale_factor = trans_scale_factor

#         self.layer_norm_s = LayerNorm(c_s)
#         self.layer_norm_z = LayerNorm(c_z)
#         self.linear_in = Linear(c_s, c_s)

#         self.ipa = InvariantPointAttention(
#             c_s, c_z, c_ipa, no_heads_ipa, no_qk_points, no_v_points, inf, epsilon
#         )
#         self.ipa_dropout = nn.Dropout(dropout_rate)
#         self.layer_norm_ipa = LayerNorm(c_s)

#         self.transition = StructureModuleTransition(c_s, no_transition_layers, dropout_rate)
#         self.bb_update = BackboneUpdate(c_s)
#         self.angle_resnet = AngleResnet(c_s, c_resnet, no_resnet_blocks, no_angles, epsilon)

#     def forward(self, evoformer_output_dict, aatype, mask=None,
#                 inplace_safe=False, _offload_inference=False, _z_reference_list=None):
#         """
#         兼容原版调用签名，额外参数仅做兼容处理
#         """
#         s = evoformer_output_dict["single"]
#         z = evoformer_output_dict["pair"]
#         B, N_res, _ = s.shape

#         if mask is None:
#             mask = s.new_ones(B, N_res)

#         # 输入归一化
#         s = self.layer_norm_s(s)
#         z = self.layer_norm_z(z)
#         s_initial = s
#         s = self.linear_in(s)

#         # 初始化单位刚体
#         rot_mats = torch.eye(3, device=s.device, dtype=s.dtype).expand(B, N_res, 3, 3).contiguous()
#         trans = torch.zeros(B, N_res, 3, device=s.device, dtype=s.dtype)

#         outputs = []
#         for i in range(self.no_blocks):
#             ipa_out = self.ipa(s, z, rot_mats, trans, mask)
            
#             # ---------- 融合路径：Linear 和 LayerNorm 均开启 Triton 时生效 ----------
#             if Linear.use_triton and LayerNorm.use_triton:
#                 B, N_res, _ = s.shape
#                 s_flat = s.reshape(B * N_res, -1)
#                 ipa_flat = ipa_out.reshape(B * N_res, -1)
                
#                 # 复用 Linear 类的转置权重缓存
#                 lin1 = self.transition.layers[0].linear_1
#                 if lin1._cached_weight_T is None or lin1._cached_weight_T.device != s.device:
#                     lin1._cached_weight_T = lin1.weight.T.contiguous()
                
#                 # 调用融合内核，同时得到 LayerNorm 输出（Transition 层残差源）和 Linear1+ReLU 输出
#                 ln_out_flat, lin1_out_flat = triton_fused_residual_ln_linear_relu(
#                     s_flat, ipa_flat,
#                     self.layer_norm_ipa.weight, self.layer_norm_ipa.bias,
#                     lin1._cached_weight_T, lin1.bias,
#                     eps=self.layer_norm_ipa.eps,
#                     allow_tf32=Linear.use_tf32
#                 )
                
#                 # 恢复形状，trans_residual 为 Transition 层的残差源，不与外层全局 s_initial 重名
#                 trans_residual = ln_out_flat.reshape(B, N_res, -1)
#                 s = lin1_out_flat.reshape(B, N_res, -1)
                
#                 # 手动执行 Transition 剩余计算：Linear2+ReLU+Linear3+残差 走双层融合内核
#                 trans_layer = self.transition.layers[0]

#                 # 预缓存两个 Linear 的转置权重
#                 lin2 = trans_layer.linear_2
#                 if lin2._cached_weight_T is None or lin2._cached_weight_T.device != s.device:
#                     lin2._cached_weight_T = lin2.weight.T.contiguous()
#                 lin3 = trans_layer.linear_3
#                 if lin3._cached_weight_T is None or lin3._cached_weight_T.device != s.device:
#                     lin3._cached_weight_T = lin3.weight.T.contiguous()

#                 B, N_res, _ = s.shape
#                 s_flat = s.reshape(B * N_res, -1)
#                 res_flat = trans_residual.reshape(B * N_res, -1)

#                 # 调用双层融合内核：Linear2 + ReLU + Linear3 + 残差
#                 s_flat = triton_fused_two_linear_residual(
#                     s_flat,
#                     lin2._cached_weight_T, lin2.bias,
#                     lin3._cached_weight_T, lin3.bias,
#                     res_flat,
#                     allow_tf32=Linear.use_tf32
#                 )
#                 s = s_flat.reshape(B, N_res, -1)

#                 s = self.transition.dropout(s)
#                 s = self.transition.layer_norm(s)
#             # ---------- 原生路径：开关未全开时回退 ----------
#             else:
#                 s = s + ipa_out
#                 s = self.ipa_dropout(s)
#                 s = self.layer_norm_ipa(s)
#                 s = self.transition(s)

#             # 更新刚体（简化：仅更新平移，保证流程完整）
#             update = self.bb_update(s)
#             trans = trans + update[..., :3] * self.trans_scale_factor

#             # 角度预测 + 坐标输出
#             unnormalized_angles, angles = self.angle_resnet(s, s_initial)
#             pred_xyz = torch.randn(B, N_res, 14, 3, device=s.device, dtype=s.dtype)
#             sidechain_frames = torch.eye(4, device=s.device, dtype=s.dtype).expand(B, N_res, 8, 4, 4)

#             preds = {
#                 "frames": torch.cat([rot_mats.flatten(-2), trans], dim=-1),
#                 "sidechain_frames": sidechain_frames,
#                 "unnormalized_angles": unnormalized_angles,
#                 "angles": angles,
#                 "positions": pred_xyz,
#                 "states": s,
#             }
#             outputs.append(preds)

#         # 堆叠所有 block 的输出
#         outputs = {k: torch.stack([d[k] for d in outputs], dim=0) for k in outputs[0].keys()}
#         outputs["single"] = s
#         return outputs