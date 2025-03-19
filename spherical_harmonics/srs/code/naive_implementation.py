from importlib import import_module

import torch

from spherical_harmonics.utils import num_irreps_projections, calculate_lastdim_num_blocks

__all__ = ["torch_spherical_harmonic", "triton_spherical_harmonic"]

BLOCK_SIZE = 64

def torch_spherical_harmonic(l: int, coords: torch.Tensor) -> torch.Tensor:
    """
    Utility function that will call the PyTorch implementation
    of a spherical harmonic order.

    This is not intended for production use, but mainly for
    sanity checking and convenience.

    Parameters
    ----------
    l : int
        Order of spherical harmonic requested.
    coords : torch.Tensor
        N-d tensor, where the last dimension should correspond
        to xyz vectors.

    Returns
    -------
    torch.Tensor
        N-d tensor of the same dimensionality as the input coordinates,
        but the size of the last dimension equal to [2 * l + 1].

    Raises
    ------
    ModuleNotFoundError
        If order of spherical harmonic requested is not found, it is
        likely not yet implemented.
    RuntimeError
        If the PyTorch implementation of the spherical harmonic is
        not found within the module.
    RuntimeError
        If the shape of the last dimension of the ``coords`` tensor
        is not equal to three.
    """
    try:
        target_module = import_module(f"spherical_harmonics.srs.code.y_{l}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Spherical harmonic order l={l} requested, but not found!"
        ) from e
    torch_func = getattr(target_module, "_torch_fwd", None)
    if not torch_func:
        raise RuntimeError(f"PyTorch implementation of l={l} not found.")
    if coords.size(-1) != 3:
        raise RuntimeError("Expects last dimension of coordinate tensor to be 3!")
    return torch_func(coords)
