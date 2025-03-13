import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards, prop_rewards, B, S, discount, 
    stride_b, stride_s, 
    BLOCK_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr  # Ensure this is a compile-time constant
):
    """
    Triton kernel for sparse reward propagation in RL environments.
    This kernel efficiently propagates sparse rewards through a batch of state transitions.
    """

    # Batch index for parallel execution
    batch_id = tl.program_id(0)

    # Compute memory offset
    offset = batch_id * stride_b  # Offset per batch
    
    # Load state transitions and rewards safely
    state_seq = tl.arange(0, SEQ_LEN, dtype=tl.int32)  # ✅ Corrected usage of tl.arange()
    mask = state_seq < SEQ_LEN

    reward_seq = tl.load(rewards + offset + state_seq, mask=mask, other=0.0)

    # Initialize propagated rewards
    tl.store(prop_rewards + offset + state_seq, reward_seq, mask=mask)

    # Parallelized backward reward propagation
    for t in range(SEQ_LEN - 2, -1, -1):  # Iterate backwards
        prev_reward = tl.load(prop_rewards + offset + t + 1, mask=(t + 1 < SEQ_LEN), other=0.0)
        updated_reward = reward_seq[t] + discount * prev_reward
        tl.store(prop_rewards + offset + t, updated_reward, mask=(t < SEQ_LEN))


def sparse_reward_propagation_triton(rewards, discount):
    """
    High-level interface for sparse reward propagation using the Triton kernel.
    """

    B, S = rewards.shape  # Batch size & sequence length

    # Allocate output tensor
    prop_rewards = torch.zeros_like(rewards)

    # Launch Triton kernel
    grid = (B,)  # Each batch runs independently
    sparse_reward_propagation_kernel[grid](
        rewards, prop_rewards, B, S, discount,
        rewards.stride(0), rewards.stride(1), 
        BLOCK_SIZE=128,  # Optimal block size
        SEQ_LEN=S  # Sequence length must be known at compile time
    )

    return prop_rewards
