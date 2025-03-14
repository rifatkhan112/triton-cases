import torch

def sparse_reward_propagation_naive(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    check_consistency: bool = False
) -> torch.Tensor:
    """
    Complete naive implementation with done flag handling and validation
    
    Args:
        rewards: Tensor of shape [B, S] containing sparse rewards
        dones: Tensor of shape [B, S] indicating episode ends (1=terminal)
        gamma: Discount factor
        check_consistency: Verify backward accumulation matches forward logic
    
    Returns:
        Tensor of shape [B, S] with properly discounted returns
    """
    B, S = rewards.shape
    returns = torch.zeros_like(rewards)
    
    for batch in range(B):
        cumulative = 0.0
        for step in reversed(range(S)):
            # Reset cumulative return at episode boundaries
            if dones[batch, step]:
                cumulative = 0.0
            
            # Current return = immediate reward + discounted future
            cumulative = rewards[batch, step] + gamma * cumulative
            returns[batch, step] = cumulative
    
    # Optional validation for algorithm correctness
    if check_consistency:
        forward_returns = _validate_forward_pass(rewards, dones, gamma)
        if not torch.allclose(returns, forward_returns, atol=1e-6):
            raise RuntimeError("Backward-forward implementation mismatch")
    
    return returns

def _validate_forward_pass(rewards, dones, gamma):
    """Alternative forward implementation for validation"""
    B, S = rewards.shape
    forward_returns = torch.zeros_like(rewards)
    
    for batch in range(B):
        cumulative = 0.0
        active = 1.0  # Mask for active episodes
        
        for step in range(S):
            # Reset cumulative return at episode starts
            if step > 0 and dones[batch, step-1]:
                cumulative = 0.0
                active = 1.0
                
            cumulative = cumulative * gamma * active + rewards[batch, step]
            forward_returns[batch, step] = cumulative
            active = 1.0 - dones[batch, step]  # Deactivate after terminal state
    
    return forward_returns
