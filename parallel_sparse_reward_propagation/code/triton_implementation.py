import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, indices_ptr, out_ptr,
    B, S, K, discount,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    thread_id = tl.arange(0, BLOCK_SIZE)

    batch_offset = batch_id * S
    indices_offset = batch_id * K

    mask = thread_id < K
    idx = tl.load(indices_ptr + indices_offset + thread_id, mask=mask, other=-1)

    valid_mask = (idx >= 0) & (idx < (S - 1)) & mask

    reward_current = tl.load(rewards_ptr + batch_offset + idx, mask=valid_mask, other=0.0)
    reward_next = tl.load(rewards_ptr + batch_offset + idx + 1, mask=valid_mask, other=0.0)

    reward_propagated = reward_current + discount * reward_next
    tl.store(out_ptr + batch_offset + idx, reward_propagated, mask=valid_mask)

class TritonSparseRewardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards, sparse_indices, discount):
        B, S = rewards.shape
        _, K = sparse_indices.shape
        out = rewards.clone()

        BLOCK_SIZE = triton.next_power_of_2(K)
        grid = (B,)

        sparse_reward_propagation_kernel[grid](
            rewards, sparse_indices, out,
            B, S, K, discount,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(sparse_indices)
        ctx.discount = discount

        return out

    @staticmethod
    def backward(ctx, grad_output):
        sparse_indices, = ctx.saved_tensors
        discount = ctx.discount

        grad_rewards = grad_output.clone()
        B, S = grad_rewards.shape
        _, K = sparse_indices.shape

        for b in range(B):
            for k in range(K):
                idx = sparse_indices[b, k]
                if idx >= 0 and idx < S - 1:
                    grad_rewards[b, idx + 1] += discount * grad_output[b, idx]

        return grad_rewards, None, None

def sparse_reward_propagation_triton(rewards, sparse_indices, discount=0.99):
    return TritonSparseRewardFunc.apply(rewards, sparse_indices, discount)
