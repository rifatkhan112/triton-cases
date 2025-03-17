import torch

def naive_softmax(x, tau):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element to avoid overflows. Softmax is invariant to this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0] / tau
    # read MN + M elements ; write MN elements
    z = x / tau - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
