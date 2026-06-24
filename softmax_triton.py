import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    rows, cols,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < cols

    # 加载行数据，边界自动处理
    x = tl.load(input_ptr + row * cols + offs, mask=mask, other=-float('inf'))

    # 数值稳定：减去行最大值
    row_max = tl.max(x, axis=0)
    x = x - row_max

    # 指数求和 + 归一化
    x_exp = tl.exp(x)
    row_sum = tl.sum(x_exp, axis=0)
    out = x_exp / row_sum

    tl.store(output_ptr + row * cols + offs, out, mask=mask)


def triton_softmax(input):
    assert input.is_cuda
    assert input.dim() == 2
    assert input.dtype == torch.float32
    assert input.is_contiguous()

    rows, cols = input.shape
    output = torch.empty_like(input)

    # 分块大小与CUDA对齐
    BLOCK_SIZE = triton.next_power_of_2(cols)
    if BLOCK_SIZE < 1024:
        BLOCK_SIZE = 1024

    grid = (rows,)
    softmax_kernel[grid](
        input, output, rows, cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output