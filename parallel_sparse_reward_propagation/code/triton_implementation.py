import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': BS}, num_warps=num_warps, num_stages=num_stages)
        for BS in [64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['S']
)
@triton.jit
def sparse_reward_propagation_kernel(
    states, rewards, out,
    stride_s_b, stride_s_s,
    stride_r_b, stride_r_s,
    stride_o_b, stride_o_s,
    discount,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Parallelized Sparse Reward Propagation Kernel.

    Each thread handles one timestep of reward propagation.
    """

    # Get batch index
    batch_id = tl.program_id(0)

    # Compute the starting index for this batch
    offset_s = batch_id * stride_s_b
    offset_r = batch_id * stride_r_b
    offset_o = batch_id * stride_o_b

    # Load state transitions and rewards
    state_seq = tl.load(states + offset_s + tl.arange(0, S), mask=tl.arange(0, S) < S, other=0.0)
    reward_seq = tl.load(rewards + offset_r + tl.arange(0, S), mask=tl.arange(0, S) < S, other=0.0)

    # Output tensor (initialize as rewards)
    prop_rewards = reward_seq

    # Parallelized backward reward propagation
    for t in range(S - 2, -1, -1):  # Iterate backwards
        prop_rewards[t] += discount * prop_rewards[t + 1]

    # Store final propagated rewards
    tl.store(out + offset_o + tl.arange(0, S), prop_rewards, mask=tl.arange(0, S) < S)


def sparse_reward_propagation_triton(states, rewards, discount=0.99):
    """
    Triton-optimized sparse reward propagation.

    Args:
        states (torch.Tensor): (B, S) State transition tensor.
        rewards (torch.Tensor): (B, S) Sparse reward tensor.
        discount (float): Discount factor for reward propagation.

    Returns:
        torch.Tensor: (B, S) Propagated rewards.
    """

    B, S = rewards.shape
    out = torch.empty_like(rewards)

    grid = (B,)  # Launch one thread block per batch

    sparse_reward_propagation_kernel[grid](
        states, rewards, out,
        states.stride(0), states.stride(1),
        rewards.stride(0), rewards.stride(1),
        out.stride(0), out.stride(1),
        discount=discount,
        S=S,
    )

    return out