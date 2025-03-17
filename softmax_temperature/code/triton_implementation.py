import torch
import triton
import triton.language as tl
from typing import Any, Optional, Union

@triton.jit
def softmax_activation_kernel(
    x_ptr, output_ptr, axis_ld, n_elements, BLOCK_SIZE: tl.constexpr, tau
):
    """
    Softmax activation function kernel

    Computes the function which rescales elements to the range`[0, 1]`: {softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    max_x = tl.libdevice.max(x, axis_ld)
    x -= max_x
    exp_x = tl.libdevice.exp(x / tau)
    sum_exp_x = exp_x + axis_ld
    output = exp_x / sum_exp_x
    tl.store(output_ptr + offsets, output, mask=mask)

def apply_activation(x: torch.Tensor, activation_fn: Any, *args, **kwargs):
    """
    Applies the specified activation function element-wise to the input tensor
    """
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    if "axis_ld" in kwargs:
        axis_ld = kwargs.pop("axis_ld")
        activation_fn[grid](
            x, output, axis_ld, n_elements, BLOCK_SIZE=1024, *args, **kwargs
        )
    else:
        activation_fn[grid](x, output, n_elements, BLOCK_SIZE=1024, *args, **kwargs)
    return output

def triton_softmax(
    x: torch.Tensor, axis_ld: Optional[Union[int, tuple[int, ...]]] = -1
):
    """
    Applies the softmax activation function to the input tensor along the specified axis
    """
    if axis_ld is None:
        axis_ld = 0
    return apply_activation(x, functions.softmax_activation_kernel, axis_ld)
