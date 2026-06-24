#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) \
                  << " (位置: " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    }

// 标准Flash Attention分块参数
const int BLOCK_M = 128;  // Br: 每个block处理的query行数
const int BLOCK_N = 32;   // Bc: 每个K/V分块的行数
const int HEAD_DIM = 64;  // 头维度，与测试用例对齐

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch, int heads, int seq_len
) {
    int tid = threadIdx.x; // 每个线程对应一行query
    int block_id = blockIdx.x;

    // 正确映射：计算当前block对应的batch、head、query分块
    int blocks_per_head = (seq_len + BLOCK_M - 1) / BLOCK_M;
    int batch_id = block_id / (heads * blocks_per_head);
    int head_id = (block_id % (heads * blocks_per_head)) / blocks_per_head;
    int q_block_id = block_id % blocks_per_head;
    int q_row = q_block_id * BLOCK_M + tid;

    // 基地址偏移
    int base = batch_id * heads * seq_len * HEAD_DIM + head_id * seq_len * HEAD_DIM;
    const float* Q_base = Q + base;
    const float* K_base = K + base;
    const float* V_base = V + base;
    float* O_base = O + base;

    // 共享内存：+1消除Bank冲突，单块占用约16KB
    __shared__ float s_K[BLOCK_N][HEAD_DIM + 1];
    __shared__ float s_V[BLOCK_N][HEAD_DIM + 1];

    // 数值稳定变量
    float m = -INFINITY;
    float l = 0.0f;
    float o[HEAD_DIM] = {0.0f};
    float scale = rsqrtf((float)HEAD_DIM);

    // 加载当前query行到寄存器
    float q_reg[HEAD_DIM];
    if (q_row < seq_len) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            q_reg[d] = Q_base[q_row * HEAD_DIM + d] * scale;
        }
    } else {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) q_reg[d] = 0.0f;
    }

    // 沿K维度分块遍历
    int k_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    for (int k_step = 0; k_step < k_blocks; k_step++) {
        int k_start = k_step * BLOCK_N;

        // 协作加载K/V分块到共享内存
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += BLOCK_M) {
            int k_row = k_start + i / HEAD_DIM;
            int k_col = i % HEAD_DIM;
            bool valid = (k_row < seq_len);
            s_K[i / HEAD_DIM][k_col] = valid ? K_base[k_row * HEAD_DIM + k_col] : 0.0f;
            s_V[i / HEAD_DIM][k_col] = valid ? V_base[k_row * HEAD_DIM + k_col] : 0.0f;
        }
        __syncthreads();

        // 计算QK点积
        float s[BLOCK_N];
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q_reg[d] * s_K[j][d];
            }
            s[j] = dot;
        }

        // 数值稳定Softmax更新
        float m_new = m;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            if (k_start + j < seq_len) m_new = fmaxf(m_new, s[j]);
        }

        float p_exp[BLOCK_N];
        float p_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            bool valid = (k_start + j < seq_len);
            p_exp[j] = valid ? expf(s[j] - m_new) : 0.0f;
            p_sum += p_exp[j];
        }

        float alpha = expf(m - m_new);
        l = l * alpha + p_sum;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            o[d] = o[d] * alpha;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j++) {
                o[d] += p_exp[j] * s_V[j][d];
            }
        }
        m = m_new;
        __syncthreads();
    }

    // 归一化写回
    if (q_row < seq_len) {
        float inv_l = 1.0f / l;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            O_base[q_row * HEAD_DIM + d] = o[d] * inv_l;
        }
    }
}

torch::Tensor cuda_flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.device().is_cuda() && K.device().is_cuda() && V.device().is_cuda());
    TORCH_CHECK(Q.dim() == 4, "Input shape must be [B, H, N, D]");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only support float32");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.size(3) == HEAD_DIM, "Head dim must be 64");

    int batch = Q.size(0);
    int heads = Q.size(1);
    int seq_len = Q.size(2);
    auto O = torch::empty_like(Q);

    int blocks_per_head = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 block(BLOCK_M);
    dim3 grid(batch * heads * blocks_per_head);

    flash_attention_kernel<<<grid, block>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
        batch, heads, seq_len
    );

    CHECK_CUDA(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_flash_attention", &cuda_flash_attention, "Fixed Standard Flash Attention v1");
}