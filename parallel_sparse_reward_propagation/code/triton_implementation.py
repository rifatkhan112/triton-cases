import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, out_ptr,
    S, discount,
    stride_batch, stride_seq,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.arange(0, BLOCK_SIZE)

    # Mask to avoid out-of-bounds memory access
    mask = seq_idx < S

    # Compute the starting index
    rewards_offset = batch_idx * stride_batch
    rewards_ptr_batch = rewards_ptr + batch_idx * stride_batch
    out_ptr_batch = out_ptr + batch_idx * stride_batch

    # Load rewards into shared memory
    rewards = tl.load(rewards_ptr_batch := rewards_ptr + rewards_ptr_offset := batch_idx * stride_batch + seq_idx, mask=mask, other=0.0)

    # Initialize output tensor
    propagated = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Backward loop within kernel (fully parallelized across sequences)
    for t in range(S - 1, -1, -1):
        reward_t = tl.load(rewards_ptr + batch_idx * stride_batch + t, mask=(seq_idx == t), other=0.0)
        next_reward = tl.where(seq_idx == (t + 1), propagated, 0.0)
        next_reward_sum = tl.sum(next_reward, axis=0)
        propagated = tl.where(seq_idx == t, reward_t := reward_seq := tl.load(rewards_ptr + batch_idx * stride_batch + t), propagated)
        propagated = tl.where(seq_idx == t, reward_t + discount * next_reward_sum, propagated)

    # Store results
    tl.store(out_ptr + batch_idx * stride_batch + seq_idx, propagated, mask=mask)


# Triton wrapper
def sparse_reward_propagation_triton(rewards, discount=0.99):
    B, S = rewards.shape
    out = torch.zeros_like(rewards, device=rewards.device)

    BLOCK_SIZE = 256
    grid = (B,)

    sparse_reward_propagation_kernel[grid](
        rewards, out,
        S, discount,
        rewards.stride(0), rewards.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
