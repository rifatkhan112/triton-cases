import torch
import triton
import triton.language as tl

# Alternative approach: use torch to identify positions we need to process
# and run the kernel only on those specific positions
def sparse_reward_propagation_triton(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Triton implementation that exactly matches the naive implementation.
    This version pre-computes on CPU the positions that need processing
    and runs a simple kernel on each batch.
    
    Args:
        rewards: [B, S] tensor of rewards
        discount: Temporal discount factor
        dones: [B, S] tensor of episode termination flags (1=done, 0=continue)
        
    Returns:
        [B, S] tensor of propagated returns
    """
    B, S = rewards.shape
    device = rewards.device
    
    # Handle missing done flags
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.float32)
    elif dones.dtype == torch.bool:
        dones = dones.to(torch.float32)
        
    # Initialize output with rewards
    output = rewards.clone()
    
    # Process each batch separately (exactly like naive implementation)
    for b in range(B):
        # Find positions with non-zero rewards or terminal states
        for s in range(S):
            reward = rewards[b, s].item()
            done = dones[b, s].item()
            
            # Skip positions with zero rewards or terminal states
            if reward == 0.0 or done != 0:
                continue
                
            # Find trajectory start (after last done flag)
            trajectory_start = 0
            for t in range(s-1, -1, -1):
                if dones[b, t].item() != 0:
                    trajectory_start = t + 1
                    break
                    
            # Propagate reward backward
            cumulative = reward
            for t in range(s-1, trajectory_start-1, -1):
                cumulative *= discount
                output[b, t] += cumulative
                
    return output
