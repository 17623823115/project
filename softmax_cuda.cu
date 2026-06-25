// softmax_cuda.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm> // for next_power_of_two

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) \
                  << " (位置: " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    }

const int MAX_BLOCK_SIZE = 1024; // 多数 GPU 最大线程数限制

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols,
    int block_size // 动态传入分块大小
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 动态共享内存：max 和 sum 各 +1 Padding 消除 Bank 冲突
    extern __shared__ float s_mem[];
    float* s_max = s_mem;
    float* s_sum = s_mem + block_size + 1;

    // 第一步：分块规约求行最大值（数值稳定）
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += block_size) {
        float val = input[row * cols + i];
        max_val = fmaxf(max_val, val);
    }
    s_max[tid] = max_val;
    __syncthreads();

    // 共享内存二分规约
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float row_max = s_max[0];

    // 第二步：计算指数和
    float sum_val = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = input[row * cols + i];
        sum_val += expf(val - row_max);
    }
    s_sum[tid] = sum_val;
    __syncthreads();

    // 规约求和
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0];

    // 第三步：计算 Softmax 并写回
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += block_size) {
        float val = input[row * cols + i];
        output[row * cols + i] = expf(val - row_max) * inv_sum;
    }
}

torch::Tensor cuda_softmax(torch::Tensor input, int block_size=-1) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Only support 2D tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only support float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);

    // 动态计算分块大小（和 Triton 对齐）
    if (block_size == -1) {
        block_size = 1;
        while (block_size < cols) block_size <<= 1; // next_power_of_2
        block_size = std::max(block_size, 1024);
    }
    block_size = std::min(block_size, MAX_BLOCK_SIZE); // 不超过 GPU 上限

    dim3 block(block_size);
    dim3 grid(rows);

    // 动态共享内存大小：(block_size +1) * 2（max + sum）
    size_t shared_mem = (block_size + 1) * 2 * sizeof(float);

    softmax_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols, block_size
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // 确保 kernel 执行完成
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_softmax", &cuda_softmax, "CUDA Softmax with adaptive block size",
          py::arg("input"), py::arg("block_size") = -1);
}