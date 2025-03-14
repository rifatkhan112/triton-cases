import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, out_ptr, S, discount, BLOCK_SIZE: tl.constexpr
):
    """
    Fully parallel Triton kernel for sparse reward propagation in reinforcement learning.
    - Each thread processes a block of time steps in reverse order.
    - Uses warp-coalesced memory access for efficient reads/writes.
    """
    pid = tl.program_id(0)  # Get batch index
    rewards_offset = pid * S  # Offset for batch processing
    seq = tl.arange(0, BLOCK_SIZE)  # Define block indices

    # Initialize the propagated reward to zero
    propagated_reward = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate backward through time in blocks
    for block_start in range(S - BLOCK_SIZE, -1, -BLOCK_SIZE):
        idx = rewards_offset + block_start + seq
        mask = (block_start + seq) < S  # Ensure we stay in bounds

        # Load rewards from memory
        rewards = tl.load(rewards_ptr + idx, mask=mask, other=0.0)

        # Accumulate rewards backward within block
        propagated_reward = rewards + discount * propagated_reward

        # Store result back in global memory
        tl.store(out_ptr + idx, propagated_reward, mask=mask)

class TritonSparseRewardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards, discount):
        B, S = rewards.shape
        out = torch.zeros_like(rewards)  # Allocate output tensor

        BLOCK_SIZE = 256
        grid = (B,)  # One thread per batch

        # Launch the Triton kernel
        sparse_reward_propagation_kernel[grid](
            rewards, out, S, discount, BLOCK_SIZE=BLOCK_SIZE
        )

        # Save necessary tensors for backward pass
        ctx.save_for_backward(rewards, out)
        ctx.discount = discount
        return out

    @staticmethod
    def backward(ctx, grad_output):
        rewards, _ = ctx.saved_tensors
        discount = ctx.discount

        grad_rewards = grad_output.clone()
        S = grad_rewards.shape[1]

        # Backward accumulation similar to naive implementation
        for t in range(S - 1):
            grad_rewards[:, t] += discount * grad_rewards[:, t + 1]

        return grad_rewards, None

def sparse_reward_propagation_triton(rewards, discount=0.99):
    """
    Entry point for Triton-based sparse reward propagation.
    """
    return TritonSparseRewardFunc.apply(rewards, discount)
