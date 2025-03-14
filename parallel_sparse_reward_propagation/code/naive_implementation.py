import torch

def sparse_reward_propagation_naive(rewards, discount):
    """
    Naive implementation of sparse reward propagation using PyTorch.

    Args:
        rewards (Tensor): shape (B, S), rewards tensor.
        discount (float): Discount factor for reward propagation.

    Returns:
        Tensor: Propagated rewards of shape (B, S).
    """
    B, S = rewards.shape
    out = rewards.clone()
    for t in reversed(range(S - 1)):
        out[:, t] += discount * out[:, t + 1]
    return out
