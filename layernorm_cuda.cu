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

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int rows, int cols,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 共享内存 +1 Padding 消Bank冲突
    __shared__ float s_sum[BLOCK_SIZE + 1];
    __shared__ float s_sq_sum[BLOCK_SIZE + 1];

    // 第一步：规约求总和、平方和
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = input[row * cols + i];
        sum += val;
        sq_sum += val * val;
    }
    s_sum[tid] = sum;
    s_sq_sum[tid] = sq_sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / cols;
    float var = s_sq_sum[0] / cols - mean * mean;
    float inv_std = rsqrtf(var + eps); // rsqrt = 1/sqrt，纯FP32计算

    // 第二步：归一化 + 仿射变换，写回
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = input[row * cols + i];
        float norm_val = (val - mean) * inv_std * gamma[i] + beta[i];
        output[row * cols + i] = norm_val;
    }
}

torch::Tensor cuda_layernorm(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Only support 2D tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only support float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gamma.size(0) == input.size(1) && beta.size(0) == input.size(1), "Gamma/beta size mismatch");

    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    layernorm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols, eps
    );

    CHECK_CUDA(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_layernorm", &cuda_layernorm, "LayerNorm v4: shared memory reduction");
}