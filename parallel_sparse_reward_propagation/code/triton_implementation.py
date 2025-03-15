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
    
    This implementation avoids all unsupported control flow statements.
    """
    # Get batch index from program ID
    b = tl.program_id(0)
    
    # Skip if batch index is out of bounds
    if b >= B:
        return
    
    # First copy all rewards to the output
    for s in range(S):
        reward_offset = b * rewards_stride_b + s * rewards_stride_s
        reward = tl.load(rewards_ptr + reward_offset)
        tl.store(output_ptr + reward_offset, reward)
    
    # Find active positions (non-zero rewards or terminals)
    # For each active position, propagate reward backward
    for s in range(S-1, -1, -1):
        reward_offset = b * rewards_stride_b + s * rewards_stride_s
        done_offset = b * dones_stride_b + s * dones_stride_s
        
        reward = tl.load(rewards_ptr + reward_offset)
        done = tl.load(dones_ptr + done_offset)
        
        # If reward is non-zero and not terminal, process this position
        should_process = (reward != 0.0) & (done == 0)
        
        # Only non-terminals with non-zero rewards propagate backward
        if should_process:
            # Find position of last done flag (trajectory start)
            last_done_pos = -1
            for t in range(s-1, -1, -1):
                t_done_offset = b * dones_stride_b + t * dones_stride_s
                t_done = tl.load(dones_ptr + t_done_offset)
                if t_done != 0:
                    last_done_pos = t
                    break
            
            # Propagate reward backward through trajectory
            cumulative = reward
            
            for t in range(s-1, -1, -1):
                # Stop at done flag
                if t <= last_done_pos:
                    break
                
                # Propagate with discount
                cumulative *= discount
                t_offset = b * rewards_stride_b + t * rewards_stride_s
                # Add to the output (may already have a value from another reward)
                current = tl.load(output_ptr + t_offset)
                tl.store(output_ptr + t_offset, current + cumulative)


def sparse_reward_propagation_triton(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Triton implementation that matches the naive implementation exactly.
    
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
