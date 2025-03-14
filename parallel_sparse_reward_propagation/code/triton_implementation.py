import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, mask_ptr, out_ptr,
    S, discount,
    stride_batch, stride_seq,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    pid = tl.program_id(1)  # Parallel across warps for long sequences
    seq_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = seq_idx < S
    
    # Load rewards and mask for this block
    rewards_block = tl.load(rewards_ptr + batch_idx * stride_batch + seq_idx, mask=mask, other=0.0)
    mask_block = tl.load(mask_ptr + batch_idx * stride_batch + seq_idx, mask=mask, other=0)
    
    # Initialize with original rewards (preserve sparse values)
    propagated = tl.where(mask_block, rewards_block, 0.0)
    
    # Backward propagation within block (sequential but vectorized)
    for offset in range(BLOCK_SIZE - 1, -1, -1):
        t = seq_idx[offset]
        if t < S - 1 and mask_block[offset] == 0:  # Only propagate non-masked
            next_val = tl.load(out_ptr + batch_idx * stride_batch + (t + 1), mask=(t + 1 < S), other=0.0)
            propagated = tl.where(seq_idx == t, discount * next_val, propagated)
    
    tl.store(out_ptr + batch_idx * stride_batch + seq_idx, propagated, mask=mask)

def sparse_reward_propagation_triton(rewards, mask, discount=0.99):
    B, S = rewards.shape
    out = rewards.clone()
    BLOCK_SIZE = 128  # Tune based on GPU
    grid = (B, triton.cdiv(S, BLOCK_SIZE))
    
    sparse_reward_propagation_kernel[grid](
        rewards, mask, out,
        S, discount,
        rewards.stride(0), rewards.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
