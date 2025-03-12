import triton
import triton.language as tl
import torch

@triton.jit
def sparse_reward_propagation_kernel(
    states_ptr, rewards_ptr, output_ptr, gamma, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for efficient sparse reward propagation."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    rewards = tl.load(rewards_ptr + offsets, mask=mask)
    output = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(BLOCK_SIZE - 2, -1, -1):
        if rewards[i] != 0:
            output[i] = rewards[i]
        else:
            output[i] = gamma * output[i + 1]

    tl.store(output_ptr + offsets, output, mask=mask)

def sparse_reward_propagation_triton(states, rewards, gamma=0.99):
    """Optimized Triton function for sparse reward propagation."""
    n_elements = rewards.numel()
    output = torch.zeros_like(rewards)

    grid = (triton.cdiv(n_elements, 1024),)
    sparse_reward_propagation_kernel[grid](
        states, rewards, output, gamma, n_elements, BLOCK_SIZE=1024
    )
    return output

if __name__ == "__main__":
    states = torch.arange(10, device='cuda')
    rewards = torch.tensor([0, 0, 1, 0, 0, 0, 2, 0, 0, 0], dtype=torch.float32, device='cuda')
    result = sparse_reward_propagation_triton(states, rewards)
    print("Triton Propagated Rewards:", result.cpu().numpy())