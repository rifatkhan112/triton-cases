import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, out_ptr,
    B, S, discount,
    BLOCK_SIZE: tl.constexpr
):
    """
    Full backward accumulation in a single Triton kernel.
    This handles multiple timesteps per thread via a loop.
    """

    batch_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)  # Parallel thread offsets

    base_idx = batch_id * S  # Global memory offset per batch
    state_idx = offsets  # Each thread maps to a sequence index
    mask = state_idx < S  # Ensure no out-of-bounds access

    # Load rewards
    rewards = tl.load(rewards_ptr + base_idx + state_idx, mask=mask, other=0.0)

    # Initialize output with rewards
    propagated = rewards

    # Full backward accumulation in the kernel
    for t in range(S - 2, -1, -1):  # Reverse order
        prev_reward = tl.load(out_ptr + base_idx + t + 1, mask=mask & (t + 1 < S), other=0.0)
        propagated = tl.where(state_idx == t, propagated + discount * prev_reward, propagated)
        tl.store(out_ptr + base_idx + t, propagated, mask=mask & (t < S))

def sparse_reward_propagation_triton(rewards, discount=0.99):
    """
    Calls the Triton kernel to propagate sparse rewards efficiently.
    """
    B, S = rewards.shape
    out = rewards.clone()
    out.copy_(rewards)  # Ensure correct initialization

    BLOCK_SIZE = 1024  # Optimize for warp efficiency
    grid = (B,)

    sparse_reward_propagation_kernel[grid](
        rewards, out,
        B, S, discount,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
