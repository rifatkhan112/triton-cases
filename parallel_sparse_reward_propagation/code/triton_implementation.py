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
    Triton kernel for efficient sparse reward propagation.
    Each thread processes a single (batch, sequence) element.
    
    Args:
        rewards_ptr, dones_ptr, output_ptr: Pointers to input/output tensors
        B, S: Batch size and sequence length
        discount: Discount factor
        rewards_stride_b, rewards_stride_s: Strides for reward tensor
        dones_stride_b, dones_stride_s: Strides for done tensor
        BLOCK_SIZE: Size of the block for parallel processing
    """
    # Get program ID and compute indices
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    
    # Process multiple elements per thread using a grid-stride loop
    for idx in range(pid, B * S, grid_size):
        # Convert linear index to batch and sequence indices
        b = idx // S
        s = idx % S
        
        # Load reward and done flag for this position
        reward_offset = b * rewards_stride_b + s * rewards_stride_s
        done_offset = b * dones_stride_b + s * dones_stride_s
        
        reward = tl.load(rewards_ptr + reward_offset)
        done = tl.load(dones_ptr + done_offset)
        
        # Only process if this position has a non-zero reward or is a terminal state
        if (reward != 0.0) | (done != 0):
            # Find trajectory start (earliest position after a done flag)
            trajectory_start = 0
            for t in range(s - 1, -1, -1):
                prev_done = tl.load(dones_ptr + b * dones_stride_b + t * dones_stride_s)
                if prev_done != 0:
                    trajectory_start = t + 1
                    break
            
            # Handle terminal state differently
            if done != 0:
                # For terminal states, just add the reward to the current position
                tl.atomic_add(output_ptr + reward_offset, reward)
            else:
                # For non-terminal rewards, propagate backward
                cumulative = reward
                tl.atomic_add(output_ptr + reward_offset, cumulative)
                
                # Propagate through trajectory with exponential decay
                for t in range(s - 1, trajectory_start - 1, -1):
                    cumulative *= discount
                    tl.atomic_add(output_ptr + b * rewards_stride_b + t * rewards_stride_s, cumulative)


def sparse_reward_propagation_triton(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Triton implementation for propagating sparse rewards through state transitions.
    Only processes elements with non-zero rewards or terminal states.
    
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
    
    # Grid size tuned for good occupancy
    grid = (triton.cdiv(B * S, 256),)
    
    # Launch kernel
    sparse_reward_propagate_kernel[grid](
        rewards, dones, output,
        B, S, discount,
        rewards.stride(0), rewards.stride(1),
        dones.stride(0), dones.stride(1),
        BLOCK_SIZE=256
    )
    
    return output
