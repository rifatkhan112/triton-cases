import torch

def sparse_reward_propagation_naive_enhanced(rewards, reward_mask=None, discount=0.99):
    B, S = rewards.shape
    out = rewards.clone()
    
    # If no mask provided, assume all rewards are sparse (original behavior)
    if reward_mask is None:
        reward_mask = torch.ones_like(rewards, dtype=torch.bool)
    
    for t in reversed(range(S - 1)):
        # Only propagate non-sparse rewards (mask=False)
        propagate = ~reward_mask[:, t]
        out[propagate, t] = discount * out[propagate, t + 1]
        
    return out
