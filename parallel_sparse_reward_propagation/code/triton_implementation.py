import triton
import triton.language as tl
import torch

@triton.jit
def sparse_reward_propagation_kernel(
    rewards, transitions, importance_weights, prop_rewards, 
    B, S, discount,
    stride_b, stride_s, BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for sparse reward propagation in RL environments.
    It efficiently propagates sparse rewards through a batch of state transitions.
    """
    
    # Batch index for parallel execution
    batch_id = tl.program_id(0)

    # Compute memory offset
    offset = batch_id * stride_b  # Offset per batch
    
    # Load state transitions and rewards safely
    state_seq = tl.arange(0, S)  # Sequential state indices
    reward_seq = tl.load(rewards + offset + state_seq, mask=state_seq < S, other=0.0)

    # Initialize propagated rewards
    tl.store(prop_rewards + offset + state_seq, reward_seq, mask=state_seq < S)

    # Parallelized backward reward propagation
    for t in range(S - 2, -1, -1):  # Iterate backwards
        prev_reward = tl.load(prop_rewards + offset + t + 1, mask=(t + 1 < S), other=0.0)
        updated_reward = reward_seq[t] + discount * prev_reward
        tl.store(prop_rewards + offset + t, updated_reward, mask=(t < S))

def sparse_reward_propagation_triton(rewards, transitions, importance_weights, discount=0.99):
    """
    Triton implementation of sparse reward propagation.

    Args:
        rewards (torch.Tensor): Shape [B, S], Sparse reward tensor.
        transitions (torch.Tensor): Shape [B, S], State transitions.
        importance_weights (torch.Tensor): Shape [B, S], Importance weights.
        discount (float): Discount factor for reward propagation.

    Returns:
        torch.Tensor: Propagated rewards tensor of shape [B, S].
    """
    B, S = rewards.shape
    prop_rewards = torch.zeros_like(rewards)

    # Launch Triton Kernel
    sparse_reward_propagation_kernel[(B,)](
        rewards, transitions, importance_weights, prop_rewards, 
        B, S, discount,
        rewards.stride(0), rewards.stride(1), 
        BLOCK_SIZE=128  # Block size for parallel execution
    )
    
    return prop_rewards
