import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, out_ptr,
    S, discount,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    seq_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Start from the end of sequence
    for block_start in range(S - BLOCK_SIZE, -1, -BLOCK_SIZE):
        idx = batch_id * S + block_start + seq_offsets
        mask = (block_start + seq_offsets) < S

        rewards = tl.load(rewards_ptr + idx, mask=mask, other=0.0)
        next_rewards = tl.load(out_ptr + idx + 1, mask=mask & (block_start + seq_offsets + 1 < S), other=0.0)

        # Perform backward accumulation
        cumulative_rewards = rewards + discount * next_rewards

        tl.store(out_ptr + idx, cumulative_rewards, mask=mask)

class TritonSparseRewardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards, discount):
        B, S = rewards.shape
        out = torch.zeros_like(rewards)

        # Initially copy rewards to output
        out.copy_(rewards)

        BLOCK_SIZE = 1024  # Optimal block size
        grid = (B,)

        # Launch kernel to perform backward accumulation
        sparse_reward_propagation_kernel[grid](
            rewards, out, S, discount, BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(torch.tensor(discount), torch.tensor(S))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        discount, S = ctx.saved_tensors
        discount = discount.item()
        S = S.item()
        
        grad_rewards = grad_output.clone()

        # Match backward gradient accumulation logic
        for t in range(S - 2, -1, -1):
            grad_rewards[:, t] += discount * grad_rewards[:, t + 1]

        return grad_rewards, None

def sparse_reward_propagation_triton(rewards, discount=0.99):
    return TritonSparseRewardFunc.apply(rewards, discount)
