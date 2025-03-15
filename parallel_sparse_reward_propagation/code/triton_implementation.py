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
    
    This implementation avoids break and continue statements.
    """
    # Get batch index from program ID
    b = tl.program_id(0)
    
    # Skip if batch index is out of bounds
    if b >= B:
        return
        
    # Initialize cumulative return
    cumulative = 0.0
    
    # Process sequence in reverse order (from end to beginning)
    for s in range(S-1, -1, -1):
        # Load reward and done flag
        reward_offset = b * rewards_stride_b + s * rewards_stride_s
        done_offset = b * dones_stride_b + s * dones_stride_s
        
        reward = tl.load(rewards_ptr + reward_offset)
        done = tl.load(dones_ptr + done_offset)
        
        # Reset cumulative value if this is a terminal state
        cumulative = tl.where(done != 0, 0.0, cumulative)
        
        # Add current reward to cumulative value
        cumulative = reward + discount * cumulative
        
        # Store result
        tl.store(output_ptr + reward_offset, cumulative)


def sparse_reward_propagation_triton(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Triton implementation that matches the naive implementation exactly.
    Each batch is processed by a dedicated thread to match the batch-wise
    processing of the naive implementation.
    
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
