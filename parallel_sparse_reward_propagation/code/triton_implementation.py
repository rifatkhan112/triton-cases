import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, prop_rewards_ptr, discount,
    B, S, stride_b, stride_s,
    BLOCK_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr
):
    """
    Optimized Triton kernel for sparse reward propagation in RL environments.
    - Uses proper indexing and broadcasting to avoid redundant memory access.
    - Avoids non-trivial loops inside Triton.
    """

    batch_id = tl.program_id(0)  # Batch index

    # Compute batch offset
    offset = batch_id * stride_b
    
    # Load rewards
    state_seq = tl.arange(0, SEQ_LEN, dtype=tl.int32)  # Triton constexpr arange
    mask = state_seq < SEQ_LEN

    reward_seq = tl.load(rewards_ptr + offset + state_seq, mask=mask, other=0.0)
    prop_rewards = reward_seq.clone()  # Initialize with current rewards

    # Reverse propagation loop (parallelized)
    for t in range(SEQ_LEN - 2, -1, -1):
        prev_reward = tl.load(prop_rewards_ptr + offset + t + 1, mask=(t + 1 < SEQ_LEN), other=0.0)
        prop_rewards[t] += discount * prev_reward
    
    # Store back
    tl.store(prop_rewards_ptr + offset + state_seq, prop_rewards, mask=mask)


def sparse_reward_propagation_triton(rewards, discount):
    """
    High-level wrapper for sparse reward propagation using the optimized Triton kernel.
    """

    B, S = rewards.shape  # Batch size & sequence length

    # Allocate output tensor
    prop_rewards = torch.zeros_like(rewards)

    # Launch Triton kernel
    grid = (B,)  # Each batch runs independently
    sparse_reward_propagation_kernel[grid](
        rewards, prop_rewards, discount,
        B, S,
        rewards.stride(0), rewards.stride(1),
        BLOCK_SIZE=128,  # Optimal block size
        SEQ_LEN=S  # Must be compile-time constant
    )

    return prop_rewards
