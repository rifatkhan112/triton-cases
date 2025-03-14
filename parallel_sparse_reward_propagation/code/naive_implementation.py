import torch

def sparse_reward_propagation_naive(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Full naive implementation with done flag handling
    Args:
        rewards: [B, S] tensor of rewards
        discount: Temporal discount factor
        dones: [B, S] tensor of episode termination flags (1=done)
    Returns:
        [B, S] tensor of propagated returns
    """
    B, S = rewards.shape
    returns = torch.zeros_like(rewards)
    
    # Handle missing done flags
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.bool)
    
    for b in range(B):
        cumulative = 0.0
        for s in reversed(range(S)):
            if dones[b, s]:
                cumulative = 0.0
            cumulative = rewards[b, s] + discount * cumulative
            returns[b, s] = cumulative
            
    return returns
