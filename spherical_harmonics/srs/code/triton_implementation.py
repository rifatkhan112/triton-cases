from importlib import import_module

import torch
import numpy as np

from spherical_harmonics.utils import num_irreps_projections, calculate_lastdim_num_blocks

__all__ = ["torch_spherical_harmonic", "triton_spherical_harmonic"]

BLOCK_SIZE = 64

def _get_autograd_func(l: int) -> type[torch.autograd.Function]:
    """
    Function that will grab the autograd.Function for a specified
    l order.

    Parameters
    ----------
    l : int
        Order of spherical harmonic to compute.

    Returns
    -------
    type[torch.autograd.Function]
        Class reference to the autograd Function.

    Raises
    ------
    ModuleNotFoundError:
        If the order of spherical harmonic is not implemented,
        the module will not exist.
    RuntimeError:
        If the autograd.Function can't be found.
    """
    try:
        target_module = import_module(f"spherical_harmonics.srs.code.y_{l}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Spherical harmonic order l={l} requested, but not found!"
        ) from e
    defined_objs = dir(target_module)
    for key in defined_objs:
        if "SphericalHarmonic" in key:
            sph_harm_func = getattr(target_module, key)
            return sph_harm_func
    raise RuntimeError(f"Namespace for module l={l} is broken!")

def triton_spherical_harmonic(
    l_values: int | list[int], coords: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Utility function that will call the Triton implementation
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
        If the Triton implementation of the spherical harmonic is
        not found within the module.
    RuntimeError
        If the shape of the last dimension of the ``coords`` tensor
        is not equal to three.
    """
    if coords.size(-1) != 3:
        raise RuntimeError("Expects last dimension of coordinate tensor to be 3!")
    if isinstance(l_values, int):
        l_values = [
            l_values,
        ]
    # ensure we are in ascending order
    l_values = list(sorted(l_values))
    dims = [num_irreps_projections(l) for l in l_values]
    offsets = np.zeros_like(dims)
    # prepend zero, since we start with zero offset
    offsets[1:] = np.cumsum(dims[:-1])

    # convert into a list, since np.int64 is not desired
    offsets = offsets.tolist()
    # preallocate a tensor that holds all of the spherical harmonic terms
    output_tensor = torch.empty(
        (*coords.shape[:-1], sum(dims)),
        device=coords.device,
        dtype=coords.dtype,
        requires_grad=True,
    )
    for l, offset in zip(l_values, offsets):
        sph_harm_func = _get_autograd_func(l)
        sph_harm_func.apply(coords, mask, BLOCK_SIZE, offset)
    return output_tensor
