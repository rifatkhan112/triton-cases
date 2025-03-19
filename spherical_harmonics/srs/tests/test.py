import pytest
import torch

from spherical_harmonics.srs.code.naive_implementation import torch_spherical_harmonic
from spherical_harmonics.srs.code.triton_implementation import triton_spherical_harmonic

device = torch.device("cuda") if torch.cuda.is_available() else None
if device is None:
    raise ValueError("CUDA is not available")

torch.manual_seed(316165)

@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("tensor_shape", [(512, 3), (128, 16, 3), (256, 8, 8, 3)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.float16,
            marks=pytest.mark.xfail(raises=AssertionError, reason="low precision"),
        ),
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(raises=AssertionError, reason="low precision"),
        ),
        torch.float32,
        torch.float64,
    ],
)

def test_forward_equivalence(order, device, tensor_shape, dtype):
    """
    This test equivalences the PyTorch and Triton implementations numerically. Its main purpose is to ensure that outputs are written back out correctly.
    """
    coords = torch.rand(tensor_shape, device, dtype=dtype)
    triton_out = triton_spherical_harmonic(order, coords)
    torch_out = torch_spherical_harmonic(order, coords)
    assert torch.allclose(triton_out, torch_out, atol=1e-5, rtol=1e-3), \
        f"""Mismatch in output for order={order}, tensor_shape={tensor_shape}, dtype={dtype},
            Output Tensor:
            {triton_out}
            Outupt Ref Tensor:
            {torch_out}
            Max differnece:
            {torch.abs(triton_out - torch_out).max()}
        """

@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("tensor_shape", [(512, 3), (128, 16, 3), (256, 8, 8, 3)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.float16,
            marks=pytest.mark.xfail(raises=AssertionError, reason="low precision"),
        ),
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(raises=AssertionError, reason="low precision"),
        ),
        torch.float32,
        torch.float64,
    ],
)

def test_backward_equivalence(order, device, tensor_shape, dtype):
    """
    Tests the numerical equivalence of the PyTorch versus the Triton implementation of the backward pass. This is mainly to ensure that writing outputs back out is being done 
    correctly.
    """
    coords = torch.rand(tensor_shape, device, dtype=dtype, requires_grad=True)
    # run with autograd first
    torch_out = torch_spherical_harmonic(order, coords)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    torch_grad = coords.grad.clone().detach()
    coords.grad.zero_()
    # now run the triton result
    triton_out = triton_spherical_harmonic(order, coords)
    triton_out.backward(gradient=torch.ones_like(triton_out))
    triton_grad = coords.grad.clone().detach()
    assert torch.allclose(triton_grad, torch_grad, atol=1e-5, rtol=1e-3), \
        f"""Mismatch in output for order={order}, tensor_shape={tensor_shape}, dtype={dtype},
            Output Tensor:
            {triton_grad}
            Outupt Ref Tensor:
            {torch_grad}
            Max differnece:
            {torch.abs(triton_grad - torch_grad).max()}
        """
