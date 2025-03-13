import torch

def sparse_reward_propagation_naive(rewards, discount=0.99):
    """
    Naive implementation of sparse reward propagation for RL environments.

    Args:
        rewards (torch.Tensor): Sparse rewards tensor of shape (B, S).
        discount (float): Discount factor for future rewards.

    Returns:
        torch.Tensor: Propagated rewards of shape (B, S).
    """
    B, S = rewards.shape  # Batch size & sequence length

    # Initialize propagated rewards
    propagated_rewards = rewards.clone()

    # Iterate backwards over time for reward propagation
    for t in reversed(range(S - 1)):  
        propagated_rewards[:, t] += discount * propagated_rewards[:, t + 1]

    return propagated_rewards
