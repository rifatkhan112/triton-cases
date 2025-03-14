import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, out_ptr,
    B, S, discount: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Parallel sparse reward propagation using warp-synchronous logic.
    Each thread updates a single state while ensuring correctness.
    """
    batch_id = tl.program_id(0)  # Each block processes one batch
    state_idx = tl.arange(0, BLOCK_SIZE)  # Parallel states

    base_idx = batch_id * S  # Offset for batch in global memory
    mask = state_idx < S  # Valid state indices

    # Load rewards into shared memory
    rewards = tl.load(rewards_ptr + base_idx + state_idx, mask=mask, other=0.0)

    # Backward accumulation inside the kernel
    for t in range(S - 2, -1, -1):  # Reverse order
        prev_reward = tl.load(out_ptr + base_idx + t + 1, mask=t + 1 < S, other=0.0)
        rewards = tl.where(state_idx == t, rewards + discount * prev_reward, rewards)
        tl.store(out_ptr + base_idx + t, rewards, mask=mask)

def sparse_reward_propagation_triton(rewards, discount=0.99):
    """
    Single-kernel sparse reward propagation using Triton.
    """
    B, S = rewards.shape
    out = rewards.clone()

    BLOCK_SIZE = min(1024, S)  # Choose block size adaptively
    grid = (B,)  # One block per batch

    sparse_reward_propagation_kernel[grid](
        rewards, out, B, S, discount, BLOCK_SIZE=BLOCK_SIZE
    )

    return out
