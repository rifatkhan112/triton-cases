import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_copy_kernel(
    src_ptr, dst_ptr,
    N,  # total number of elements: B*S
    BLOCK_SIZE: tl.constexpr
):
    """
    Simple Triton kernel that copies 'src' into 'dst' elementwise in parallel.
    No dynamic loops. 
    """
    pid = tl.program_id(0)   # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, val, mask=mask)

def sparse_reward_propagation_triton(rewards, discount=0.99):
    """
    1) Copies 'rewards' into 'propagated_rewards' using Triton kernel (elementwise).
    2) Applies the backward pass in Python (like the naive approach).
    """
    B, S = rewards.shape
    N = B * S  # total elements

    # Allocate output
    propagated_rewards = torch.zeros_like(rewards)

    # 1) Triton kernel: elementwise copy of `rewards` -> `propagated_rewards`
    BLOCK_SIZE = 256
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1D grid
    elementwise_copy_kernel[grid](
        rewards, propagated_rewards,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # 2) Perform backward pass in Python
    #    (same logic as naive_implementation, for fair comparison)
    for t in reversed(range(S - 1)):
        propagated_rewards[:, t] += discount * propagated_rewards[:, t + 1]

    return propagated_rewards
