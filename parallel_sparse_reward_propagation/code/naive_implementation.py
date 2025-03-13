import torch

def sparse_reward_propagation_naive(states, rewards, discount=0.99):
    """
    Naive CPU-based implementation of sparse reward propagation.

    Args:
        states (torch.Tensor): Batch of state transitions (B, S).
        rewards (torch.Tensor): Sparse reward tensor (B, S).
        discount (float): Discount factor for reward propagation.

    Returns:
        torch.Tensor: Propagated rewards.
    """
    B, S, *_ = rewards.shape  # Allows extra dimensions
    propagated_rewards = rewards.clone()

    # Iterate backwards to propagate rewards
    for t in range(S - 2, -1, -1):  # From second-last timestep to first
        propagated_rewards[:, t] += discount * propagated_rewards[:, t + 1]

    return propagated_rewards
