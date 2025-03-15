import torch

def sparse_reward_propagation_naive(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Naive implementation for propagating sparse rewards through state transitions.
    Efficiently handles sparsity by only processing non-zero rewards and terminal states.
    
    Args:
        rewards: [B, S] tensor of rewards
        discount: Temporal discount factor
        dones: [B, S] tensor of episode termination flags (1=done, 0=continue)
        
    Returns:
        [B, S] tensor of propagated returns
    """
    B, S = rewards.shape
    returns = torch.zeros_like(rewards)
    
    # Handle missing done flags
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.bool)
    elif dones.dtype != torch.bool:
        dones = dones.bool()
    
    # First identify positions with non-zero rewards or terminal states
    active_positions = []
    for b in range(B):
        for s in range(S):
            if rewards[b, s] != 0 or dones[b, s]:
                active_positions.append((b, s))
    
    # Process each active position
    for b, s in active_positions:
        # Find the start of this trajectory segment (after the last done flag)
        trajectory_start = 0
        for t in range(s-1, -1, -1):
            if dones[b, t]:
                trajectory_start = t + 1
                break
        
        # Propagate the reward backward through the trajectory
        if dones[b, s]:
            # For terminal states, only add the reward at the current position
            returns[b, s] += rewards[b, s]
        else:
            # For non-terminal states, propagate the reward backward
            cumulative = rewards[b, s]
            # Add to current position
            returns[b, s] += cumulative
            
            # Propagate backwards with discount
            for t in range(s-1, trajectory_start-1, -1):
                cumulative *= discount
                returns[b, t] += cumulative
    
    return returns
