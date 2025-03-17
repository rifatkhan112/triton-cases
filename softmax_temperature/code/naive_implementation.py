import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 1 #########
# first we'll look at the naive implementation jic you need a refresher
def naive_softmax(x):
    '''
    Built for input of size (M,N)
    Safe softmax is when we subtract the maximum element in order to avoid numerical 
    overflows when doing .exp(); softmax is invariant to this shift
    '''
    # read MN elements, find their max along N, and write M elements (the maxes)
    x_max = x.max(dim=1)[0] 
        # pytorch actually outputs a tuple of (values, indices) so [0] grabs the values;
        # we ignored the indices when talking about memory writes above
    # read MN + M elements, subtraction is MN flops, and write MN elements
    tau = 100
    z = (x - x_max[:, None]) / tau
    # read MN elements and write MN elemnts
    numerator = torch.exp(z)
        # exp is actually a lot of flops per element but we're only worried about mem ops rn
    # read MN elements, do MN flops to find M sum values, and then write M elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements, division is MN flops, then write MN elements
    out = numerator / denominator[:, None]

    # in total we did 8MN + 4M memory operations
    # (read 5MN + 2M elements; wrote 3MN + 2M elements)
    return out
