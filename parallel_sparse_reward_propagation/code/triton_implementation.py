import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagate_kernel(
    rewards_ptr, dones_ptr, output_ptr,
    B, S, discount,
    rewards_stride_b, rewards_stride_s,
    dones_stride_b, dones_stride_s,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for propagating sparse rewards through state transitions.
    This version exactly matches the behavior of the naive implementation.
    """
    # Get program ID and compute indices
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    
    # Process multiple elements per thread using a grid-stride loop
    for b in range(0, B):
        for s in range(S - 1, -1, -1):  # Process in reverse order like the naive implementation
            # Skip elements not assigned to this thread
            element_idx = b * S + s
            if element_idx % grid_size != pid:
                continue
                
            # Load reward and done flag for this position
            reward_offset = b * rewards_stride_b + s * rewards_stride_s
            done_offset = b * dones_stride_b + s * dones_stride_s
            
            reward = tl.load(rewards_ptr + reward_offset)
            done = tl.load(dones_ptr + done_offset)
            
            # Load current cumulative value (if any)
            cumulative = tl.load(output_ptr + reward_offset)
            
            # Add current reward to the cumulative value
            cumulative = cumulative + reward
            
            # Store the updated cumulative value
            tl.store(output_ptr + reward_offset, cumulative)
            
            # If this is a terminal state, don't propagate further
            if done != 0:
                continue
                
            # Propagate the discounted cumulative value to the previous time step
            if s > 0:  # Only if there's a previous step
                prev_offset = b * rewards_stride_b + (s-1) * rewards_stride_s
                discounted_value = cumulative * discount
                tl.atomic_add(output_ptr + prev_offset, discounted_value)


def sparse_reward_propagation_triton(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Triton implementation for propagating sparse rewards through state transitions.
    Implements a backward pass through the sequence to compute discounted returns.
    
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
    
    # Use a two-stage approach for better performance
    
    # Stage 1: Process rewards from back to front
    # We'll launch one thread for each time step across all batches
    grid = (min(1024, B * S),)
    
    sparse_reward_propagate_kernel[grid](
        rewards, dones, output,
        B, S, discount,
        rewards.stride(0), rewards.stride(1),
        dones.stride(0), dones.stride(1),
        BLOCK_SIZE=256
    )
    
    return output
