import torch

def sparse_reward_propagation_naive(rewards, discount=0.99):
    """
    Naive Python-based implementation of backward reward propagation.
    Performs an in-place backward pass on CPU/GPU using standard PyTorch ops.
    Args:
        rewards (torch.Tensor): (B, S) tensor of rewards.
        discount (float): Scalar discount factor.
    Returns:
        torch.Tensor: (B, S) in-place updated for backward pass.
    """
    B, S = rewards.shape
    # Clone so we don't overwrite the original
    propagated_rewards = rewards.clone()

    # Backward pass in Python
    for t in reversed(range(S - 1)):
        propagated_rewards[:, t] += discount * propagated_rewards[:, t + 1]

    return propagated_rewards
