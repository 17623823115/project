#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) \
                  << " (位置: " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    }

const int BLOCK_SIZE = 1024;

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 共享内存 +1 Padding 彻底消除Bank冲突
    __shared__ float s_max[BLOCK_SIZE + 1];
    __shared__ float s_sum[BLOCK_SIZE + 1];

    // 第一步：分块规约求行最大值（数值稳定）
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = input[row * cols + i];
        max_val = fmaxf(max_val, val);
    }
    s_max[tid] = max_val;
    __syncthreads();

    // 共享内存二分规约
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float row_max = s_max[0];

    // 第二步：计算指数和
    float sum_val = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = input[row * cols + i];
        sum_val += expf(val - row_max);
    }
    s_sum[tid] = sum_val;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0];

    // 第三步：计算softmax并写回
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = input[row * cols + i];
        output[row * cols + i] = expf(val - row_max) * inv_sum;
    }
}

torch::Tensor cuda_softmax(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Only support 2D tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only support float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    softmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols
    );

    CHECK_CUDA(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_softmax", &cuda_softmax, "Softmax v4: shared memory reduction, bank conflict free");
}