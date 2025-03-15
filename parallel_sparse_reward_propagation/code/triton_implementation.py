import torch
import triton
import triton.language as tl

@triton.jit
def reverse_rewards_kernel(
    rewards_ptr, dones_ptr, tmp_buffer_ptr,
    B, S, discount,
    rewards_stride_b, rewards_stride_s,
    dones_stride_b, dones_stride_s,
    BLOCK_SIZE: tl.constexpr
):
    """
    First stage: Store rewards and dones in reverse order in a temporary buffer.
    This makes the forward processing in the second stage equivalent to
    backward processing in the naive implementation.
    """
    pid = tl.program_id(0)
    
    # Each thread handles one batch
    if pid < B:
        b = pid
        
        # Process sequence in reverse order and store in temp buffer
        for s in range(S):
            orig_s = S - 1 - s  # Original position (reversed)
            
            # Load from original position
            reward = tl.load(rewards_ptr + b * rewards_stride_b + orig_s * rewards_stride_s)
            done = tl.load(dones_ptr + b * dones_stride_b + orig_s * dones_stride_s)
            
            # Store in temporary buffer at forward position
            tmp_offset = b * S * 2 + s * 2
            tl.store(tmp_buffer_ptr + tmp_offset, reward)
            tl.store(tmp_buffer_ptr + tmp_offset + 1, done)


@triton.jit
def process_rewards_kernel(
    tmp_buffer_ptr, output_ptr,
    B, S, discount,
    output_stride_b, output_stride_s,
    BLOCK_SIZE: tl.constexpr
):
    """
    Second stage: Process the reversed rewards to compute returns.
    The forward processing of reversed data is equivalent to 
    backward processing of the original data.
    """
    pid = tl.program_id(0)
    
    # Each thread handles one batch
    if pid < B:
        b = pid
        
        # Initialize cumulative return
        cumulative = 0.0
        
        # Process sequence in forward order (which is backward in original data)
        for s in range(S):
            # Load from temp buffer
            tmp_offset = b * S * 2 + s * 2
            reward = tl.load(tmp_buffer_ptr + tmp_offset)
            done = tl.load(tmp_buffer_ptr + tmp_offset + 1)
            
            # Reset cumulative value at terminal states
            cumulative = tl.where(done != 0, 0.0, cumulative)
            
            # Add current reward to cumulative value
            cumulative = reward + discount * cumulative
            
            # Store result at the original position (reversing back)
            orig_s = S - 1 - s
            output_offset = b * output_stride_b + orig_s * output_stride_s
            tl.store(output_ptr + output_offset, cumulative)


def sparse_reward_propagation_triton(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """
    Triton implementation that matches the naive implementation exactly.
    Uses a two-stage approach with temporary storage to handle the backward propagation.
    
    Args:
        rewards: [B, S] tensor of rewards
        discount: Temporal discount factor
        dones: [B, S] tensor of episode termination flags (1=done, 0=continue)
        
    Returns:
        [B, S] tensor of propagated returns
    """
    B, S = rewards.shape
    device = rewards.device
    output = torch.zeros_like(rewards)
    
    # Handle missing done flags
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.float32)
    elif dones.dtype == torch.bool:
        dones = dones.to(torch.float32)
    
    # Create temporary buffer to store reversed data
    # Format: [batch][position][reward/done]
    tmp_buffer = torch.zeros(B * S * 2, device=device, dtype=torch.float32)
    
    # Stage 1: Reverse the data
    grid1 = (B,)
    reverse_rewards_kernel[grid1](
        rewards, dones, tmp_buffer,
        B, S, discount,
        rewards.stride(0), rewards.stride(1),
        dones.stride(0), dones.stride(1),
        BLOCK_SIZE=1
    )
    
    # Stage 2: Process the reversed data
    grid2 = (B,)
    process_rewards_kernel[grid2](
        tmp_buffer, output,
        B, S, discount,
        output.stride(0), output.stride(1),
        BLOCK_SIZE=1
    )
    
    return output
