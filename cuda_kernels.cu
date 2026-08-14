#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ===================== LayerNorm  =====================
// 128
__global__ void layernorm_128_kernel(
    const float* input, const float* gamma, const float* beta,
    float* output, int rows, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float sum_val[128];
    __shared__ float sum_sq_val[128];
    
    float x = input[row * 128 + tid];
    sum_val[tid] = x;
    sum_sq_val[tid] = x * x;
    __syncthreads();
    
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) {
            sum_val[tid] += sum_val[tid + s];
            sum_sq_val[tid] += sum_sq_val[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sum_val[0] * 0.0078125f;
    float var = sum_sq_val[0] * 0.0078125f - mean * mean;
    float inv_std = rsqrt(var + eps);
    
    float g = gamma[tid];
    float b = beta[tid];
    output[row * 128 + tid] = (x - mean) * inv_std * g + b;
}

// 384
__global__ void layernorm_384_kernel(
    const float* input, const float* gamma, const float* beta,
    float* output, int rows, float eps) {
    const int COLS = 384;
    const float inv_cols = 1.0f / 384.0f;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float sum_val[512];
    __shared__ float sum_sq_val[512];
    
    float sum = 0.0f, sum_sq = 0.0f;
    if (tid < COLS) {
        float x = input[row * COLS + tid];
        sum = x;
        sum_sq = x * x;
    }
    sum_val[tid] = sum;
    sum_sq_val[tid] = sum_sq;
    __syncthreads();
    
    for (int s = 256; s > 0; s >>= 1) {
        if (tid < s) {
            sum_val[tid] += sum_val[tid + s];
            sum_sq_val[tid] += sum_sq_val[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sum_val[0] * inv_cols;
    float var = sum_sq_val[0] * inv_cols - mean * mean;
    float inv_std = rsqrt(var + eps);
    
    if (tid < COLS) {
        float x = input[row * COLS + tid];
        float g = gamma[tid];
        float b = beta[tid];
        output[row * COLS + tid] = (x - mean) * inv_std * g + b;
    }
}

// normal
template<int BLOCK_SIZE>
__global__ void layernorm_generic_kernel(
    const float* input, const float* gamma, const float* beta,
    float* output, int rows, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float sum_val[BLOCK_SIZE];
    __shared__ float sum_sq_val[BLOCK_SIZE];
    
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float x = input[row * cols + i];
        sum += x;
        sum_sq += x * x;
    }
    
    sum_val[tid] = sum;
    sum_sq_val[tid] = sum_sq;
    __syncthreads();
    
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_val[tid] += sum_val[tid + s];
            sum_sq_val[tid] += sum_sq_val[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sum_val[0] / cols;
    float var = sum_sq_val[0] / cols - mean * mean;
    float inv_std = rsqrt(var + eps);
    
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float x = input[row * cols + i];
        float g = gamma[i];
        float b = beta[i];
        output[row * cols + i] = (x - mean) * inv_std * g + b;
    }
}

// ===================== Softmax  =====================
template<int BLOCK_N>
__global__ void softmax_row_kernel(
    const float* input, float* output, int M, int N, int stride_row) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float max_val[BLOCK_N];
    __shared__ float sum_exp[BLOCK_N];
    
    float max_f = -INFINITY;
    for (int i = tid; i < N; i += BLOCK_N) {
        max_f = max(max_f, input[row * stride_row + i]);
    }
    max_val[tid] = max_f;
    __syncthreads();
    for (int s = BLOCK_N / 2; s > 0; s >>= 1) {
        if (tid < s) max_val[tid] = max(max_val[tid], max_val[tid + s]);
        __syncthreads();
    }
    float row_max = max_val[0];
    
    float sum_f = 0.0f;
    for (int i = tid; i < N; i += BLOCK_N) {
        sum_f += exp(input[row * stride_row + i] - row_max);
    }
    sum_exp[tid] = sum_f;
    __syncthreads();
    for (int s = BLOCK_N / 2; s > 0; s >>= 1) {
        if (tid < s) sum_exp[tid] += sum_exp[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sum_exp[0];
    
    for (int i = tid; i < N; i += BLOCK_N) {
        output[row * stride_row + i] = exp(input[row * stride_row + i] - row_max) * inv_sum;
    }
}

template<int BLOCK_N>
__global__ void softmax_row_masked_kernel(
    const float* input, const float* mask, float* output,
    int M, int N, int stride_row, int stride_mask_row) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float max_val[BLOCK_N];
    __shared__ float sum_exp[BLOCK_N];
    
    float max_f = -INFINITY;
    for (int i = tid; i < N; i += BLOCK_N) {
        float x = input[row * stride_row + i] + mask[row * stride_mask_row + i];
        max_f = max(max_f, x);
    }
    max_val[tid] = max_f;
    __syncthreads();
    for (int s = BLOCK_N / 2; s > 0; s >>= 1) {
        if (tid < s) max_val[tid] = max(max_val[tid], max_val[tid + s]);
        __syncthreads();
    }
    float row_max = max_val[0];
    
    float sum_f = 0.0f;
    for (int i = tid; i < N; i += BLOCK_N) {
        float x = input[row * stride_row + i] + mask[row * stride_mask_row + i];
        sum_f += exp(x - row_max);
    }
    sum_exp[tid] = sum_f;
    __syncthreads();
    for (int s = BLOCK_N / 2; s > 0; s >>= 1) {
        if (tid < s) sum_exp[tid] += sum_exp[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sum_exp[0];
    
    for (int i = tid; i < N; i += BLOCK_N) {
        float x = input[row * stride_row + i] + mask[row * stride_mask_row + i];
        output[row * stride_row + i] = exp(x - row_max) * inv_sum;
    }
}

//  GEMM basic configuration (pure CUDA core optimized version, aligned with Triton non-tensor core optimization) 
#define GEMM_BLOCK_M    64    // The M-dimensional block size of the thread block
#define GEMM_BLOCK_N    64    // The N-dimensional block size of the thread block
#define GEMM_BLOCK_K    32    // The K-dimensional block size of the thread block
#define GEMM_THREAD_M   4     // Single-threaded calculation of the number of elements in the M direction (register block)
#define GEMM_THREAD_N   4     // Single-threaded calculation of the number of elements in the N direction (register block)
#define GEMM_GROUP_M    4     // M-dimensional group scheduling, aligning Triton GROUP_M, enhancing L2 cache reuse

#define GEMM_BLK_DIM_M  (GEMM_BLOCK_M / GEMM_THREAD_M)
#define GEMM_BLK_DIM_N  (GEMM_BLOCK_N / GEMM_THREAD_N)

// =====================  GEMM + Bias + ReLU  =====================
__global__ void gemm_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    bool has_relu)
{
    // 1. One-dimensional Block mapping to two-dimensional partitioning + GROUP_M grouping scheduling (alignment of Triton GROUP_M optimization)
    int block_id = blockIdx.x;
    int num_blocks_n = (N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;
    int num_blocks_m = (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M;
    int num_blocks_per_group = GEMM_GROUP_M * num_blocks_n;
    int group_id = block_id / num_blocks_per_group;
    int group_inner_id = block_id % num_blocks_per_group;
    int block_m = group_id * GEMM_GROUP_M + group_inner_id % GEMM_GROUP_M;
    int block_n = group_inner_id / GEMM_GROUP_M;

    if (block_m >= num_blocks_m || block_n >= num_blocks_n) return;

    int g_m_base = block_m * GEMM_BLOCK_M;
    int g_n_base = block_n * GEMM_BLOCK_N;

    // 2. Thread-level register partitioning: Each thread computes 4×4 outputs, maximizing register reuse
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int thread_m_start = g_m_base + tid_y * GEMM_THREAD_M;
    int thread_n_start = g_n_base + tid_x * GEMM_THREAD_N;

    float acc[GEMM_THREAD_M][GEMM_THREAD_N];
    #pragma unroll
    for (int i = 0; i < GEMM_THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < GEMM_THREAD_N; j++)
            acc[i][j] = 0.0f;

    // 3. Shared memory double buffering (Ping-Pong pipelining), aligning Triton num_stages multi-stage optimization
    __shared__ float sA[2][GEMM_BLOCK_M][GEMM_BLOCK_K];
    __shared__ float sB[2][GEMM_BLOCK_K][GEMM_BLOCK_N];

    int tid_linear = tid_y * GEMM_BLK_DIM_N + tid_x;
    int stage = 0;

    // Preload the first K-dimensional data block
    #pragma unroll
    for (int i = tid_linear; i < GEMM_BLOCK_M * GEMM_BLOCK_K; i += blockDim.x * blockDim.y) {
        int row = i / GEMM_BLOCK_K;
        int col = i % GEMM_BLOCK_K;
        int g_row = g_m_base + row;
        int g_col = col;
        sA[stage][row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
    }
    #pragma unroll
    for (int i = tid_linear; i < GEMM_BLOCK_K * GEMM_BLOCK_N; i += blockDim.x * blockDim.y) {
        int row = i / GEMM_BLOCK_N;
        int col = i % GEMM_BLOCK_N;
        int g_row = row;
        int g_col = g_n_base + col;
        sB[stage][row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
    }
    __syncthreads();

    // 4. K-dimensional main loop: Computation parallel to off-chip memory preloading, masking memory access latency
    #pragma unroll 1
    for (int k = 0; k < K; k += GEMM_BLOCK_K) {
        int compute_stage = stage;
        int next_stage = 1 - stage;
        int next_k = k + GEMM_BLOCK_K;

        // Preload the next batch of data (executed concurrently with the current calculation)
        if (next_k < K) {
            #pragma unroll
            for (int i = tid_linear; i < GEMM_BLOCK_M * GEMM_BLOCK_K; i += blockDim.x * blockDim.y) {
                int row = i / GEMM_BLOCK_K;
                int col = i % GEMM_BLOCK_K;
                int g_row = g_m_base + row;
                int g_col = next_k + col;
                sA[next_stage][row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
            }
            #pragma unroll
            for (int i = tid_linear; i < GEMM_BLOCK_K * GEMM_BLOCK_N; i += blockDim.x * blockDim.y) {
                int row = i / GEMM_BLOCK_N;
                int col = i % GEMM_BLOCK_N;
                int g_row = next_k + row;
                int g_col = g_n_base + col;
                sB[next_stage][row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
            }
        }

        // Current block-based computing: Full expansion of sub-loops, enhancement of instruction-level parallelism
        #pragma unroll
        for (int kk = 0; kk < GEMM_BLOCK_K; kk++) {
            float a_reg[GEMM_THREAD_M];
            float b_reg[GEMM_THREAD_N];

            #pragma unroll
            for (int i = 0; i < GEMM_THREAD_M; i++) {
                a_reg[i] = sA[compute_stage][tid_y * GEMM_THREAD_M + i][kk];
            }
            #pragma unroll
            for (int j = 0; j < GEMM_THREAD_N; j++) {
                b_reg[j] = sB[compute_stage][kk][tid_x * GEMM_THREAD_N + j];
            }

            #pragma unroll
            for (int i = 0; i < GEMM_THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < GEMM_THREAD_N; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
        stage = next_stage;
    }

    // 5. The Bias + ReLU operation is completed within the register, without any additional memory read/write operations.
    float bias_reg[GEMM_THREAD_N];
    #pragma unroll
    for (int j = 0; j < GEMM_THREAD_N; j++) {
        int col = thread_n_start + j;
        bias_reg[j] = (col < N && bias != nullptr) ? bias[col] : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < GEMM_THREAD_M; i++) {
        int row = thread_m_start + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < GEMM_THREAD_N; j++) {
            int col = thread_n_start + j;
            if (col >= N) continue;

            float val = acc[i][j] + bias_reg[j];
            if (has_relu) val = max(val, 0.0f);
            C[row * N + col] = val;
        }
    }
}

// =====================  GEMM + Bias + ReLU + Residual fusion kernel =====================
__global__ void gemm_bias_residual_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ C,
    int M, int N, int K,
    bool has_relu)
{
    // 1. One-dimensional Block mapping + GROUP_M Group scheduling
    int block_id = blockIdx.x;
    int num_blocks_n = (N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;
    int num_blocks_m = (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M;
    int num_blocks_per_group = GEMM_GROUP_M * num_blocks_n;
    int group_id = block_id / num_blocks_per_group;
    int group_inner_id = block_id % num_blocks_per_group;
    int block_m = group_id * GEMM_GROUP_M + group_inner_id % GEMM_GROUP_M;
    int block_n = group_inner_id / GEMM_GROUP_M;

    if (block_m >= num_blocks_m || block_n >= num_blocks_n) return;

    int g_m_base = block_m * GEMM_BLOCK_M;
    int g_n_base = block_n * GEMM_BLOCK_N;

    // 2. Thread-level register partitioning
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int thread_m_start = g_m_base + tid_y * GEMM_THREAD_M;
    int thread_n_start = g_n_base + tid_x * GEMM_THREAD_N;

    float acc[GEMM_THREAD_M][GEMM_THREAD_N];
    #pragma unroll
    for (int i = 0; i < GEMM_THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < GEMM_THREAD_N; j++)
            acc[i][j] = 0.0f;

    // 3. Shared memory double buffering
    __shared__ float sA[2][GEMM_BLOCK_M][GEMM_BLOCK_K];
    __shared__ float sB[2][GEMM_BLOCK_K][GEMM_BLOCK_N];

    int tid_linear = tid_y * GEMM_BLK_DIM_N + tid_x;
    int stage = 0;

    // Preload the first block
    #pragma unroll
    for (int i = tid_linear; i < GEMM_BLOCK_M * GEMM_BLOCK_K; i += blockDim.x * blockDim.y) {
        int row = i / GEMM_BLOCK_K;
        int col = i % GEMM_BLOCK_K;
        int g_row = g_m_base + row;
        int g_col = col;
        sA[stage][row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
    }
    #pragma unroll
    for (int i = tid_linear; i < GEMM_BLOCK_K * GEMM_BLOCK_N; i += blockDim.x * blockDim.y) {
        int row = i / GEMM_BLOCK_N;
        int col = i % GEMM_BLOCK_N;
        int g_row = row;
        int g_col = g_n_base + col;
        sB[stage][row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
    }
    __syncthreads();

    // 4. K-dimensional main loop
    #pragma unroll 1
    for (int k = 0; k < K; k += GEMM_BLOCK_K) {
        int compute_stage = stage;
        int next_stage = 1 - stage;
        int next_k = k + GEMM_BLOCK_K;

        // Preload the next piece
        if (next_k < K) {
            #pragma unroll
            for (int i = tid_linear; i < GEMM_BLOCK_M * GEMM_BLOCK_K; i += blockDim.x * blockDim.y) {
                int row = i / GEMM_BLOCK_K;
                int col = i % GEMM_BLOCK_K;
                int g_row = g_m_base + row;
                int g_col = next_k + col;
                sA[next_stage][row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
            }
            #pragma unroll
            for (int i = tid_linear; i < GEMM_BLOCK_K * GEMM_BLOCK_N; i += blockDim.x * blockDim.y) {
                int row = i / GEMM_BLOCK_N;
                int col = i % GEMM_BLOCK_N;
                int g_row = next_k + row;
                int g_col = g_n_base + col;
                sB[next_stage][row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
            }
        }

        // Current block calculation
        #pragma unroll
        for (int kk = 0; kk < GEMM_BLOCK_K; kk++) {
            float a_reg[GEMM_THREAD_M];
            float b_reg[GEMM_THREAD_N];

            #pragma unroll
            for (int i = 0; i < GEMM_THREAD_M; i++) {
                a_reg[i] = sA[compute_stage][tid_y * GEMM_THREAD_M + i][kk];
            }
            #pragma unroll
            for (int j = 0; j < GEMM_THREAD_N; j++) {
                b_reg[j] = sB[compute_stage][kk][tid_x * GEMM_THREAD_N + j];
            }

            #pragma unroll
            for (int i = 0; i < GEMM_THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < GEMM_THREAD_N; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
        stage = next_stage;
    }

    // 5. Register-based Bias + ReLU + Residual Superposition
    float bias_reg[GEMM_THREAD_N];
    #pragma unroll
    for (int j = 0; j < GEMM_THREAD_N; j++) {
        int col = thread_n_start + j;
        bias_reg[j] = (col < N && bias != nullptr) ? bias[col] : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < GEMM_THREAD_M; i++) {
        int row = thread_m_start + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < GEMM_THREAD_N; j++) {
            int col = thread_n_start + j;
            if (col >= N) continue;

            float val = acc[i][j] + bias_reg[j];
            if (has_relu) val = max(val, 0.0f);
            val += residual[row * N + col];
            C[row * N + col] = val;
        }
    }
}

// ===================== Four-component fusion: Residual + LN + Linear + ReLU =====================
__global__ void fused_residual_ln_linear_relu_kernel(
    const float* s_ptr, const float* ipa_ptr,
    const float* ln_gamma, const float* ln_beta,
    const float* lin_weight, const float* lin_bias,
    float* ln_out_ptr, float* lin_out_ptr,
    int M, int K, int N_OUT, float eps) {
    
    const int LN_BLOCK_K = 128;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float sum_val[128];
    __shared__ float sum_sq_val[128];
    
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = tid; i < K; i += LN_BLOCK_K) {
        float x = s_ptr[row * K + i] + ipa_ptr[row * K + i];
        sum += x;
        sum_sq += x * x;
    }
    
    sum_val[tid] = sum;
    sum_sq_val[tid] = sum_sq;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) {
            sum_val[tid] += sum_val[tid + s];
            sum_sq_val[tid] += sum_sq_val[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sum_val[0] / K;
    float var = sum_sq_val[0] / K - mean * mean;
    float inv_std = rsqrt(var + eps);
    
    for (int i = tid; i < K; i += LN_BLOCK_K) {
        float x = s_ptr[row * K + i] + ipa_ptr[row * K + i];
        ln_out_ptr[row * K + i] = (x - mean) * inv_std * ln_gamma[i] + ln_beta[i];
    }
    __syncthreads();
    
    for (int n = 0; n < N_OUT; n++) {
        float sum_lin = 0.0f;
        for (int i = tid; i < K; i += LN_BLOCK_K) {
            sum_lin += ln_out_ptr[row * K + i] * lin_weight[i * N_OUT + n];
        }
        sum_val[tid] = sum_lin;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) sum_val[tid] += sum_val[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            lin_out_ptr[row * N_OUT + n] = max(sum_val[0] + lin_bias[n], 0.0f);
        }
        __syncthreads();
    }
}

// ===================== Double-layer Linear + Residual Fusion Operator (for handling stack overflow) =====================
__global__ void fused_two_linear_residual_kernel(
    const float* x_ptr,
    const float* w1, const float* b1,
    const float* w2, const float* b2,
    const float* residual,
    float* out_ptr,
    int M, int K1, int K2, int N) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int BLOCK = 128;
    
    __shared__ float sum_val[128];
    __shared__ float hidden[1024]; // Move to shared memory and support large-dimensional intermediate layers
    
    for (int k = 0; k < K2; k++) {
        float sum_h = 0.0f;
        for (int i = tid; i < K1; i += BLOCK) {
            sum_h += x_ptr[row * K1 + i] * w1[i * K2 + k];
        }
        sum_val[tid] = sum_h;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) sum_val[tid] += sum_val[tid + s];
            __syncthreads();
        }
        if (tid == 0) hidden[k] = max(sum_val[0] + b1[k], 0.0f);
        __syncthreads();
    }
    
    for (int n = 0; n < N; n++) {
        float sum_o = 0.0f;
        for (int i = tid; i < K2; i += BLOCK) {
            sum_o += hidden[i] * w2[i * N + n];
        }
        sum_val[tid] = sum_o;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) sum_val[tid] += sum_val[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            out_ptr[row * N + n] = sum_val[0] + b2[n] + residual[row * N + n];
        }
        __syncthreads();
    }
}

// ===================== Point coordinate squared distance operator =====================
__global__ void point_sq_dist_kernel(
    const float* q_pts, const float* k_pts, float* dist,
    int M, int N, int P) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int BLOCK_N = 128;
    
    float q_sum = 0.0f;
    for (int p = 0; p < P; p++) {
        int base = row * P * 3 + p * 3;
        float qx = q_pts[base + 0];
        float qy = q_pts[base + 1];
        float qz = q_pts[base + 2];
        q_sum += qx*qx + qy*qy + qz*qz;
    }
    
    for (int j = 0; j < N; j += BLOCK_N) {
        int j_idx = j + tid;
        if (j_idx < N) {
            float k_sum = 0.0f;
            float dot_sum = 0.0f;
            for (int p = 0; p < P; p++) {
                int k_base = j_idx * P * 3 + p * 3;
                float kx = k_pts[k_base + 0];
                float ky = k_pts[k_base + 1];
                float kz = k_pts[k_base + 2];
                k_sum += kx*kx + ky*ky + kz*kz;
                
                int q_base = row * P * 3 + p * 3;
                dot_sum += q_pts[q_base + 0] * kx
                         + q_pts[q_base + 1] * ky
                         + q_pts[q_base + 2] * kz;
            }
            dist[row * N + j_idx] = q_sum + k_sum - 2.0f * dot_sum;
        }
    }
}

// ===================== Integrated Attention Softmax Operator (for fixing array out-of-bounds errors) =====================
__global__ void fused_attention_softmax_kernel(
    const float* q, const float* k,
    const float* extra_att, const float* mask,
    float* out, int M, int N, int C, float scale) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int BLOCK_N = 128;
    
    __shared__ float max_val[128];
    __shared__ float sum_exp[128];
    __shared__ float q_vec[1024]; // Move to shared memory and support large feature dimensions
    
    for (int i = tid; i < C; i += 128) {
        q_vec[i] = q[row * C + i];
    }
    __syncthreads();
    
    float max_f = -INFINITY;
    for (int j = 0; j < N; j += BLOCK_N) {
        int j_idx = j + tid;
        if (j_idx < N) {
            float dot = 0.0f;
            for (int c = 0; c < C; c++) {
                dot += q_vec[c] * k[j_idx * C + c];
            }
            dot = dot * scale + extra_att[row * N + j_idx] + mask[row * N + j_idx];
            max_f = max(max_f, dot);
        }
    }
    
    max_val[tid] = max_f;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) max_val[tid] = max(max_val[tid], max_val[tid + s]);
        __syncthreads();
    }
    float row_max = max_val[0];
    
    float sum_f = 0.0f;
    for (int j = 0; j < N; j += BLOCK_N) {
        int j_idx = j + tid;
        if (j_idx < N) {
            float dot = 0.0f;
            for (int c = 0; c < C; c++) {
                dot += q_vec[c] * k[j_idx * C + c];
            }
            dot = dot * scale + extra_att[row * N + j_idx] + mask[row * N + j_idx];
            sum_f += exp(dot - row_max);
        }
    }
    
    sum_exp[tid] = sum_f;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sum_exp[tid] += sum_exp[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sum_exp[0];
    
    for (int j = 0; j < N; j += BLOCK_N) {
        int j_idx = j + tid;
        if (j_idx < N) {
            float dot = 0.0f;
            for (int c = 0; c < C; c++) {
                dot += q_vec[c] * k[j_idx * C + c];
            }
            dot = dot * scale + extra_att[row * N + j_idx] + mask[row * N + j_idx];
            out[row * N + j_idx] = exp(dot - row_max) * inv_sum;
        }
    }
}

// ===================== Linear + Rigid body transformation fusion operator =====================
__global__ void fused_linear_point_transform_kernel(
    const float* x, const float* weight, const float* bias,
    const float* rot, const float* trans,
    float* out, int M, int K, int HP) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int BLOCK = 128;
    
    __shared__ float sum_val[128];
    
    float r00 = rot[row * 9 + 0], r01 = rot[row * 9 + 1], r02 = rot[row * 9 + 2];
    float r10 = rot[row * 9 + 3], r11 = rot[row * 9 + 4], r12 = rot[row * 9 + 5];
    float r20 = rot[row * 9 + 6], r21 = rot[row * 9 + 7], r22 = rot[row * 9 + 8];
    float t0 = trans[row * 3 + 0], t1 = trans[row * 3 + 1], t2 = trans[row * 3 + 2];
    
    for (int p = 0; p < HP; p++) {
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        for (int i = tid; i < K; i += BLOCK) {
            float xv = x[row * K + i];
            px += xv * weight[i * HP * 3 + p * 3 + 0];
            py += xv * weight[i * HP * 3 + p * 3 + 1];
            pz += xv * weight[i * HP * 3 + p * 3 + 2];
        }
        
        sum_val[tid] = px;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) sum_val[tid] += sum_val[tid + s];
            __syncthreads();
        }
        if (tid == 0) px = sum_val[0] + bias[p * 3 + 0];
        __syncthreads();
        
        sum_val[tid] = py;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) sum_val[tid] += sum_val[tid + s];
            __syncthreads();
        }
        if (tid == 0) py = sum_val[0] + bias[p * 3 + 1];
        __syncthreads();
        
        sum_val[tid] = pz;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) sum_val[tid] += sum_val[tid + s];
            __syncthreads();
        }
        if (tid == 0) pz = sum_val[0] + bias[p * 3 + 2];
        __syncthreads();
        
        if (tid == 0) {
            float ox = px * r00 + py * r10 + pz * r20 + t0;
            float oy = px * r01 + py * r11 + pz * r21 + t1;
            float oz = px * r02 + py * r12 + pz * r22 + t2;
            out[row * HP * 3 + p * 3 + 0] = ox;
            out[row * HP * 3 + p * 3 + 1] = oy;
            out[row * HP * 3 + p * 3 + 2] = oz;
        }
        __syncthreads();
    }
}

// ===================== C++ external interface =====================
// LayerNorm
torch::Tensor cuda_layernorm_128(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto output = torch::empty_like(input);
    int rows = input.numel() / 128;
    dim3 grid(rows);
    dim3 block(128);
    layernorm_128_kernel<<<grid, block>>>(
        input.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), rows, eps
    );
    return output;
}

torch::Tensor cuda_layernorm_384(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto output = torch::empty_like(input);
    int rows = input.numel() / 384;
    dim3 grid(rows);
    dim3 block(512);
    layernorm_384_kernel<<<grid, block>>>(
        input.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), rows, eps
    );
    return output;
}

torch::Tensor cuda_layernorm_generic(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    dim3 grid(rows);
    dim3 block(1024);
    layernorm_generic_kernel<1024><<<grid, block>>>(
        input.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), rows, cols, eps
    );
    return output;
}

// Softmax
torch::Tensor cuda_softmax(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int M = input.numel() / input.size(-1);
    int N = input.size(-1);
    dim3 grid(M);
    dim3 block(1024);
    softmax_row_kernel<1024><<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        M, N, input.stride(-2)
    );
    return output;
}

torch::Tensor cuda_softmax_masked(torch::Tensor input, torch::Tensor mask) {
    auto output = torch::empty_like(input);
    int M = input.numel() / input.size(-1);
    int N = input.size(-1);
    dim3 grid(M);
    dim3 block(1024);
    softmax_row_masked_kernel<1024><<<grid, block>>>(
        input.data_ptr<float>(), mask.data_ptr<float>(), output.data_ptr<float>(),
        M, N, input.stride(-2), mask.stride(-2)
    );
    return output;
}

// GEMM
// ===================== GEMM external interface =====================
torch::Tensor cuda_gemm_bias_relu(torch::Tensor A, torch::Tensor B, torch::Tensor bias, bool has_relu) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    int num_blocks_m = (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M;
    int num_blocks_n = (N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;
    int total_blocks = num_blocks_m * num_blocks_n;

    dim3 grid(total_blocks);
    dim3 block(GEMM_BLK_DIM_N, GEMM_BLK_DIM_M); // 16x16=256

    gemm_bias_relu_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        has_relu
    );
    return C;
}

torch::Tensor cuda_gemm_bias_residual(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor residual, bool has_relu) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    int num_blocks_m = (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M;
    int num_blocks_n = (N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;
    int total_blocks = num_blocks_m * num_blocks_n;

    dim3 grid(total_blocks);
    dim3 block(GEMM_BLK_DIM_N, GEMM_BLK_DIM_M);

    gemm_bias_residual_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        residual.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        has_relu
    );
    return C;
}

// Fusion operator
std::tuple<torch::Tensor, torch::Tensor> cuda_fused_residual_ln_linear(
    torch::Tensor s, torch::Tensor ipa,
    torch::Tensor ln_gamma, torch::Tensor ln_beta,
    torch::Tensor lin_weight, torch::Tensor lin_bias,
    float eps) {
    int M = s.size(0);
    int K = s.size(1);
    int N_OUT = lin_weight.size(1);
    auto ln_out = torch::empty_like(s);
    auto lin_out = torch::empty({M, N_OUT}, s.options());
    dim3 grid(M);
    dim3 block(128);
    fused_residual_ln_linear_relu_kernel<<<grid, block>>>(
        s.data_ptr<float>(), ipa.data_ptr<float>(),
        ln_gamma.data_ptr<float>(), ln_beta.data_ptr<float>(),
        lin_weight.data_ptr<float>(), lin_bias.data_ptr<float>(),
        ln_out.data_ptr<float>(), lin_out.data_ptr<float>(),
        M, K, N_OUT, eps
    );
    return std::make_tuple(ln_out, lin_out);
}

torch::Tensor cuda_fused_two_linear_residual(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2,
    torch::Tensor residual) {
    int M = x.size(0);
    int K1 = x.size(1);
    int K2 = w1.size(1);
    int N = w2.size(1);
    auto out = torch::empty({M, N}, x.options());
    dim3 grid(M);
    dim3 block(128);
    fused_two_linear_residual_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w1.data_ptr<float>(), b1.data_ptr<float>(),
        w2.data_ptr<float>(), b2.data_ptr<float>(),
        residual.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K1, K2, N
    );
    return out;
}

// Distance from the point
torch::Tensor cuda_point_sq_dist(torch::Tensor q_pts, torch::Tensor k_pts) {
    int M = q_pts.size(0);
    int P = q_pts.size(1);
    int N = k_pts.size(0);
    auto dist = torch::empty({M, N}, q_pts.options());
    dim3 grid(M);
    dim3 block(128);
    point_sq_dist_kernel<<<grid, block>>>(
        q_pts.data_ptr<float>(), k_pts.data_ptr<float>(), dist.data_ptr<float>(),
        M, N, P
    );
    return dist;
}

// Fusion attention
torch::Tensor cuda_fused_attention_softmax(
    torch::Tensor q, torch::Tensor k,
    torch::Tensor extra_att, torch::Tensor mask,
    float scale) {
    int M = q.size(0);
    int C = q.size(1);
    int N = k.size(0);
    auto out = torch::empty({M, N}, q.options());
    dim3 grid(M);
    dim3 block(128);
    fused_attention_softmax_kernel<<<grid, block>>>(
        q.data_ptr<float>(), k.data_ptr<float>(),
        extra_att.data_ptr<float>(), mask.data_ptr<float>(),
        out.data_ptr<float>(), M, N, C, scale
    );
    return out;
}

// Rigid body transformation fusion
torch::Tensor cuda_fused_linear_point_transform(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor rot, torch::Tensor trans, int HP) {
    int M = x.size(0);
    int K = x.size(1);
    auto out = torch::empty({M, HP, 3}, x.options());
    dim3 grid(M);
    dim3 block(128);
    fused_linear_point_transform_kernel<<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        rot.data_ptr<float>(), trans.data_ptr<float>(),
        out.data_ptr<float>(), M, K, HP
    );
    return out;
}

// ===================== Python binding =====================
PYBIND11_MODULE(cuda_kernels, m) {
    m.def("cuda_layernorm_128", &cuda_layernorm_128);
    m.def("cuda_layernorm_384", &cuda_layernorm_384);
    m.def("cuda_layernorm_generic", &cuda_layernorm_generic);
    m.def("cuda_softmax", &cuda_softmax);
    m.def("cuda_softmax_masked", &cuda_softmax_masked);
    m.def("cuda_gemm_bias_relu", &cuda_gemm_bias_relu);
    m.def("cuda_gemm_bias_residual", &cuda_gemm_bias_residual);
    m.def("cuda_fused_residual_ln_linear", &cuda_fused_residual_ln_linear);
    m.def("cuda_fused_two_linear_residual", &cuda_fused_two_linear_residual);
    m.def("cuda_point_sq_dist", &cuda_point_sq_dist);
    m.def("cuda_fused_attention_softmax", &cuda_fused_attention_softmax);
    m.def("cuda_fused_linear_point_transform", &cuda_fused_linear_point_transform);
}