#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) \
                  << " (位置: " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    }

// ========== 分块配置（平衡性能与共享内存占用，绝对不会超限） ==========
const int BLOCK_SIZE_M = 64;
const int BLOCK_SIZE_N = 64;
const int BLOCK_SIZE_K = 32;
const int THREAD_M = 8;    // 每个线程计算8x8个输出元素
const int THREAD_N = 8;

__global__ void gemm_kernel_v4_fixed(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int g_row_start = blockIdx.y * BLOCK_SIZE_M + ty * THREAD_M;
    int g_col_start = blockIdx.x * BLOCK_SIZE_N + tx * THREAD_N;

    // 寄存器累加器
    float acc[THREAD_M][THREAD_N] = {0.0f};

    // ========== 修复：共享内存定义在核函数内部，每个block独立一份 ==========
    // Padding仅+1，足够消除32路bank冲突，大幅节省共享内存
    __shared__ float s_A[BLOCK_SIZE_M][BLOCK_SIZE_K + 1];
    __shared__ float s_B[BLOCK_SIZE_K][BLOCK_SIZE_N + 1];

    int k_blocks = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    int load_idx = ty * blockDim.x + tx;
    int thread_num = blockDim.x * blockDim.y;

    for (int k_step = 0; k_step < k_blocks; k_step++) {
        int k_start = k_step * BLOCK_SIZE_K;

        // ========== 标量加载，无对齐风险，全尺寸兼容 ==========
        // 加载s_A：线程均匀分工
        int total_load_A = BLOCK_SIZE_M * BLOCK_SIZE_K;
        for (int i = load_idx; i < total_load_A; i += thread_num) {
            int s_row = i / BLOCK_SIZE_K;
            int s_col = i % BLOCK_SIZE_K;
            int g_row = blockIdx.y * BLOCK_SIZE_M + s_row;
            int g_col = k_start + s_col;
            s_A[s_row][s_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // 加载s_B
        int total_load_B = BLOCK_SIZE_K * BLOCK_SIZE_N;
        for (int i = load_idx; i < total_load_B; i += thread_num) {
            int s_row = i / BLOCK_SIZE_N;
            int s_col = i % BLOCK_SIZE_N;
            int g_row = k_start + s_row;
            int g_col = blockIdx.x * BLOCK_SIZE_N + s_col;
            s_B[s_row][s_col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        // ========== 寄存器内乘加，全循环展开 ==========
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            float a_reg[THREAD_M];
            #pragma unroll
            for (int m = 0; m < THREAD_M; m++) {
                a_reg[m] = s_A[ty * THREAD_M + m][k];
            }

            float b_reg[THREAD_N];
            #pragma unroll
            for (int n = 0; n < THREAD_N; n++) {
                b_reg[n] = s_B[k][tx * THREAD_N + n];
            }

            #pragma unroll
            for (int m = 0; m < THREAD_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_N; n++) {
                    acc[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
    #pragma unroll
    for (int m = 0; m < THREAD_M; m++) {
        int g_row = g_row_start + m;
        #pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
            int g_col = g_col_start + n;
            if (g_row < M && g_col < N) {
                C[g_row * N + g_col] = acc[m][n];
            }
        }
    }
}

// ========== PyTorch 调用接口 ==========
torch::Tensor cuda_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only support 2D matrix");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimension mismatch");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only support float32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    // 线程块尺寸：(64/8, 64/8) = (8, 8)，共64个线程
    dim3 block(BLOCK_SIZE_N / THREAD_N, BLOCK_SIZE_M / THREAD_M);
    dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_kernel_v4_fixed<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    CHECK_CUDA(cudaGetLastError());
    return C;
}

// ========== Python 绑定 ==========
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_gemm", &cuda_gemm, "Fixed GEMM v4: bank conflict free, safe shared memory");
}