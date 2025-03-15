import torch
import triton
import triton.language as tl

@triton.jit
def process_batch_kernel(
    rewards_ptr, dones_ptr, output_ptr,
    B, S, discount,
    rewards_stride_b, rewards_stride_s,
    dones_stride_b, dones_stride_s
):
    """
    Process a single batch to compute returns.
    Each kernel instance handles one batch.
    
    This implementation exactly matches the naive implementation behavior.
    """
    # Get batch index from program ID
    b = tl.program_id(0)
    
    # Skip if batch index is out of bounds
    if b >= B:
        return
    
    # Initialize values
    for s in range(S):
        # Set initial values to original rewards
        reward_offset = b * rewards_stride_b + s * rewards_stride_s
        reward = tl.load(rewards_ptr + reward_offset)
        tl.store(output_ptr + reward_offset, reward)
    
    # Process each position with non-zero reward or terminal
    for s in range(S):
        # Load reward and done flag
        reward_offset = b * rewards_stride_b + s * rewards_stride_s
        done_offset = b * dones_stride_b + s * dones_stride_s
        
        reward = tl.load(rewards_ptr + reward_offset)
        done = tl.load(dones_ptr + done_offset)
        
        # Skip processing if this is a terminal state
        is_terminal = done != 0
        
        # Only process non-terminal states with non-zero rewards
        if (reward != 0.0) & (~is_terminal):
            # Find trajectory start
            start = 0
            for t in range(s-1, -1, -1):
                # Check for done flag
                t_done = tl.load(dones_ptr + b * dones_stride_b + t * dones_stride_s)
                start = tl.where(t_done != 0, t+1, start)
            
            # Propagate reward backward through trajectory
            cumulative = reward
            
            # Propagate through trajectory with exponential decay
            for t in range(s-1, start-1, -1):
                cumulative *= discount
                t_offset = b * rewards_stride_b + t * rewards_stride_s
                tl.atomic_add(output_ptr + t_offset, cumulative)


def sparse_reward_propagation_triton(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Triton implementation that exactly matches the naive implementation.
    Each batch is processed by a dedicated thread.
    
    Args:
        rewards: [B, S] tensor of rewards
        discount: Temporal discount factor
        dones: [B, S] tensor of episode termination flags (1=done, 0=continue)
        
    Returns:
        [B, S] tensor of propagated returns
    """
    B, S = rewards.shape
    output = torch.zeros_like(rewards)
    
    # Handle missing done flags
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.float32)
    elif dones.dtype == torch.bool:
        dones = dones.to(torch.float32)
    
    # Launch one kernel per batch
    grid = (B,)
    
    process_batch_kernel[grid](
        rewards, dones, output,
        B, S, discount,
        rewards.stride(0), rewards.stride(1),
        dones.stride(0), dones.stride(1)
    )
    
    return output
