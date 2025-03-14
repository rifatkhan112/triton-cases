import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, out_ptr, B, S, discount, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)  # Batch index
    seq = tl.arange(0, BLOCK_SIZE)  # Sequence indices
    offset = pid * S  # Batch offset

    # Allocate shared memory for cumulative rewards within block
    cumulative_reward = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Start from the end of the sequence and accumulate rewards backwards
    for block_start in range(S - BLOCK_SIZE, -1, -BLOCK_SIZE):
        idx = offset + block_start + seq
        mask = (block_start + seq) >= 0  # Ensure valid indices

        # Load current rewards
        rewards = tl.load(rewards_ptr + idx, mask=mask, other=0.0)

        # Perform backward accumulation
        for i in range(BLOCK_SIZE - 1, -1, -1):
            cumulative_reward = tl.where(
                seq == i,
                rewards + discount * cumulative_reward,
                cumulative_reward
            )

        # Store cumulative reward
        tl.store(out_ptr + idx, cumulative_reward, mask=mask)

class TritonSparseRewardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards, discount):
        B, S = rewards.shape
        out = torch.zeros_like(rewards)

        BLOCK_SIZE = 256  # Define optimal block size
        grid = (B,)  # One grid per batch

        sparse_reward_propagation_kernel[grid](
            rewards, out, B, S, discount, BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(out, torch.tensor(discount))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, discount = ctx.saved_tensors
        grad_rewards = grad_output.clone()
        S = grad_rewards.shape[1]

        # Accumulate gradients backward to match naive implementation
        for t in range(S - 1):
            grad_rewards[:, t] += discount * grad_rewards[:, t + 1]

        return grad_rewards, None

def sparse_reward_propagation_triton(rewards, discount=0.99):
    return TritonSparseRewardFunc.apply(rewards, discount)
