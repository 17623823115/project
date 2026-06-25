# softmax_triton.py
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

def triton_softmax(input, block_size=None):
    """
    Args:
        input: CUDA 上的 2D float32 contiguous tensor
        block_size: 固定分块大小（None 则自适应，和原逻辑一致）
    """
    assert input.is_cuda, "Input must be CUDA tensor"
    assert input.dim() == 2, "Only support 2D tensor"
    assert input.dtype == torch.float32, "Only support float32"
    assert input.is_contiguous(), "Input must be contiguous"

    rows, cols = input.shape
    output = torch.empty_like(input)

    # 分块大小策略：自适应 / 固定
    if block_size is None:
        BLOCK_SIZE = triton.next_power_of_2(cols)
        BLOCK_SIZE = max(BLOCK_SIZE, 1024)  # 最小 1024
    else:
        BLOCK_SIZE = block_size

    grid = (rows,)
    softmax_kernel[grid](
        input, output, rows, cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output