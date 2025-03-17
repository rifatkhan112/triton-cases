import torch
import triton
import triton.language as tl
from softmax_temperature.code.naive_implementation import naive_softmax
from softmax_temperature.code.triton_implementation import triton_softmax
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
tau = 100
y_triton = triton_softmax(x, tau)
y_naive = naive_softmax(x, tau)
assert torch.allclose(y_triton, y_naive), (y_triton, y_naive)
