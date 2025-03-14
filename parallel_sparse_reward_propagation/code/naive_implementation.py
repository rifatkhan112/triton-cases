import torch

def improved_sparse_propagate(rewards: torch.Tensor, 
                            dones: torch.Tensor, 
                            gamma: float) -> torch.Tensor:
    """
    Combined batched implementation with done handling
    Args:
        rewards: Tensor of shape [B, S]
        dones: Tensor of shape [B, S] (1s at episode ends)
        gamma: Discount factor
    Returns:
        Tensor of shape [B, S] with propagated rewards
    """
    B, S = rewards.shape
    returns = torch.zeros_like(rewards)
    
    # Start from terminal state
    returns[:, -1] = rewards[:, -1]
    
    # Backward pass with episode boundary awareness
    for t in reversed(range(S-1)):
        # Mask for active episodes
        active = ~dones[:, t]
        returns[:, t] = rewards[:, t] + gamma * returns[:, t+1] * active
        
    return returns
