import torch

from softmax_temperature.tests.test import softmax

torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
tau = 100
y_triton = softmax(x, tau)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
