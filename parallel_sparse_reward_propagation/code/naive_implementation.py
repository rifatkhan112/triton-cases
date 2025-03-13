import torch

def sparse_reward_propagation_naive(rewards, transitions=None, importance_weights=None, discount=0.99):
    """
    Optimized naive implementation of sparse reward propagation for RL environments.

    Args:
        rewards (torch.Tensor): Sparse rewards tensor of shape (B, S).
        transitions (torch.Tensor, optional): Sparse transition matrix (B, S, S) or None.
        importance_weights (torch.Tensor, optional): Importance weights (B, S) or None.
        discount (float or torch.Tensor): Discount factor for future rewards (scalar or (B,) shape).

    Returns:
        torch.Tensor: Propagated rewards of shape (B, S).
    """
    B, S = rewards.shape  # Batch size & sequence length

    # Ensure discount is correctly shaped (support for per-batch discounting)
    if isinstance(discount, torch.Tensor) and discount.shape != (B,):
        raise ValueError("Discount tensor must have shape (B,) for batch-wise discounting.")

    # Initialize propagated rewards
    propagated_rewards = rewards.clone()

    # Iterate backwards over time for reward propagation
    for t in reversed(range(S - 1)):  
        # If discount is per-batch, apply correct broadcasting
        discount_factor = discount.view(B, 1) if isinstance(discount, torch.Tensor) else discount
        propagated_rewards[:, t] += discount_factor * propagated_rewards[:, t + 1]

    # Apply importance weights if provided
    if importance_weights is not None:
        propagated_rewards *= importance_weights  # Apply element-wise scaling

    # If a transition matrix is provided, apply transition-based propagation
    if transitions is not None:
        propagated_rewards = torch.bmm(transitions, propagated_rewards.unsqueeze(-1)).squeeze(-1)

    return propagated_rewards
