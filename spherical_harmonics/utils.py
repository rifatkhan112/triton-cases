import math
import torch

def num_irreps_projections(l: int) -> int:
    """
    Calculate the number of projections for a given order
    of spherical harmonic.

    Parameters
    ----------
    l : int
        Order of spherical harmonic.

    Returns
    -------
    int
        Number of projections, i.e. 2l + 1
    """
    return 2 * l + 1

def calculate_lastdim_num_blocks(input_tensor: torch.Tensor, block_size: int) -> int:
    """
    Calculate the number of blocks for a tensor, assuming we
    stride along the last dimension, and a given block size.

    The corresponding pointer arithmetic looks like this:

    ```python
    block_id = tl.program_id(0)
    striding = tl.arange(0, block_size) * stride
    offset = (striding + (block_size * stride * block_id))
    ```

    This function is used to work out the amount of parallel
    work that needs to be done, given as the total number of
    elements divided by the last dimension stride, and a specified
    block size that will then divvy up the work.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Torch N-d tensor to operate over.

    Returns
    -------
    int
        Number of blocks of work, given a block size.
    """
    # get the stride of the last dimension
    stride = input_tensor.stride(-2)
    numel = input_tensor.numel()
    total_blocks = math.ceil(numel / stride)
    return total_blocks
