import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, out_ptr, S, discount, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)  # Batch index
    offset = pid * S
    seq = tl.arange(0, BLOCK_SIZE)

    # Initialize output buffer to store cumulative rewards
    cumulative_reward = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Start from the end of the sequence and move backwards
    for block_start in range(S, 0, -BLOCK_SIZE):
        idx = offset + block_start - BLOCK_SIZE + seq
        mask = (block_start - BLOCK_SIZE + seq) >= 0

        # Load current reward block
        current_rewards = tl.load(rewards_ptr + idx, mask=mask, other=0.0)

        # Reverse accumulation within the block
        for i in range(BLOCK_SIZE - 1, -1, -1):
            cumulative_reward = tl.where(
                seq == i,
                current_rewards + discount * cumulative_reward,
                cumulative_reward
            )

        # Store cumulative reward
        tl.store(out_ptr + idx, cumulative_reward, mask=mask)

class TritonSparseRewardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards, discount):
        B, S = rewards.shape
        out = torch.zeros_like(rewards)

        BLOCK_SIZE = 256
        grid = (B,)

        sparse_reward_propagation_kernel[grid](
            rewards, out, S, discount, BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(torch.tensor(discount))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        discount, = ctx.saved_tensors
        grad_rewards = grad_output.clone()
        S = grad_rewards.shape[1]

        # Accumulate gradients backward to match naive implementation
        for t in range(S - 1):
            grad_rewards[:, t] += discount * grad_rewards[:, t + 1]

        return grad_rewards, None

def sparse_reward_propagation_triton(rewards, discount=0.99):
    return TritonSparseRewardFunc.apply(rewards, discount)
