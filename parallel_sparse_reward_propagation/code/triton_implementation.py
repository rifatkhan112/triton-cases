import torch
import triton
import triton.language as tl

@triton.jit
def reward_propagation_kernel(rewards_ptr, out_ptr, S, discount, BLOCK_SIZE: tl.constexpr):
    batch_id = tl.program_id(0)
    offset = batch_id * S
    seq = tl.arange(0, BLOCK_SIZE)
    propagated_reward = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Backward pass accumulation in parallel blocks
    for t in range(S - BLOCK_SIZE, -1, -BLOCK_SIZE):
        mask = (seq + t) < S
        reward_block = tl.load(rewards + batch_id * S + seq + t, mask=mask, other=0.0)
        propagated_reward = reward_block + discount * propagated_reward
        tl.store(out_ptr + batch_id * S + seq + t, propagated_reward, mask=mask)

@triton.jit
def sparse_reward_propagation_kernel(rewards_ptr, out_ptr, S, discount, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    rewards_offset = pid * S
    seq = tl.arange(0, BLOCK_SIZE)

    for block_start in range(S - BLOCK_SIZE, -1, -BLOCK_SIZE):
        idx = rewards_offset + block_start + seq
        mask = (block_start + seq) < S
        rewards = tl.load(rewards_ptr + rewards_offset + block_start + seq, mask=mask, other=0.0)

        # Accumulate rewards backward within block
        for i in reversed(range(BLOCK_SIZE - 1)):
            rewards[i] += discount * rewards[i + 1]

        tl.store(out_ptr + rewards_offset + block_start, rewards, mask=mask)

class TritonSparseRewardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards, discount):
        B, S = rewards.shape
        out = torch.zeros_like(rewards)

        BLOCK_SIZE = 256
        grid = (rewards.shape[0],)

        sparse_reward_propagation_kernel[grid](
            rewards, out, S, discount, BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(rewards, out)
        ctx.discount = discount
        return out

    @staticmethod
    def backward(ctx, grad_output):
        rewards, _ = ctx.saved_tensors
        discount = ctx.discount

        grad_rewards = grad_output.clone()
        S = grad_rewards.shape[1]

        for t in range(S - 1):
            grad_rewards[:, t] += discount * grad_rewards[:, t + 1]

        return grad_rewards, None

def sparse_reward_propagation_triton(rewards, discount=0.99):
    return TritonSparseRewardFunc.apply(rewards, discount)
