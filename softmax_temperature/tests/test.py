import torch
import triton
import triton.language as tl
from softmax_temperature.code.naive_implementation import naive_softmax
from softmax_temperature.code.triton_implementation import triton_softmax
from triton.runtime import driver

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    Here is where we test the wrapper function and kernel that we wrote 
    above to ensure all our values are correct, using pytorch as the 
    correct answer to compare against

    we'll use an irregular number of rows & cols to verify that our padding mechanism works
    """
    # create input data
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)
    # run kernel & pytorch reference implementation
    z_tri = triton_softmax(x)
    z_ref = naive_softmax(x)
        # notice our implementation doesn't give a choice for what axis to softmax along.
        # this is a common theme of custom GPU kernels; because pytorch has to write code that
        #  is more general, it is slower than it could be
    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")

if __name__ == "__main__":
    # always run unit-tests
    test_softmax_kernel(size=(1823, 781))
