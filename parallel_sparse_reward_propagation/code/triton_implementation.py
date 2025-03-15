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
    
    This implementation avoids ALL unsupported control flow statements (break, continue).
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
    
    # For each position, propagate reward backward if needed
    for s in range(S-1, -1, -1):
        reward_offset = b * rewards_stride_b + s * rewards_stride_s
        done_offset = b * dones_stride_b + s * dones_stride_s
        
        reward = tl.load(rewards_ptr + reward_offset)
        done = tl.load(dones_ptr + done_offset)
        
        # Only non-terminals with non-zero rewards propagate backward
        should_process = (reward != 0.0) & (done == 0)
        
        if should_process:
            # Find position of last done flag (trajectory start)
            # Use mask-based approach instead of break
            last_done_pos = -1
            found_done = False
            
            for t in range(s-1, -1, -1):
                t_done_offset = b * dones_stride_b + t * dones_stride_s
                t_done = tl.load(dones_ptr + t_done_offset)
                
                # Update last_done_pos if we find a done flag and haven't found one yet
                update_pos = (t_done != 0) & (~found_done)
                last_done_pos = tl.where(update_pos, t, last_done_pos)
                
                # Mark that we've found a done flag
                found_done = found_done | (t_done != 0)
            
            # Propagate reward backward through trajectory
            cumulative = reward
            
            for t in range(s-1, -1, -1):
                # Skip steps before trajectory start (after a done flag)
                is_valid_step = t > last_done_pos
                
                if is_valid_step:  # Only process valid steps
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
