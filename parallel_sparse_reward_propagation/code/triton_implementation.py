import torch
import triton
import triton.language as tl

@triton.jit
def sparse_reward_propagation_kernel(
    rewards_ptr, indices_ptr, out_ptr,
    B, S, K, discount,
    BLOCK_SIZE: tl.constexpr
):
    """
    Parallel implementation of sparse reward propagation using Triton.
    
    Each thread handles a sparse reward index and propagates rewards backward.
    """
    batch_id = tl.program_id(0)  # Each block handles a batch
    thread_id = tl.arange(0, BLOCK_SIZE)  # Each thread handles a separate state

    # Get the start position for this batch
    base_idx = batch_id * S  # Offset for batch in global memory
    indices_start = batch_id * K  # Offset for indices

    # Load the sparse indices for this batch
    sparse_indices = tl.load(indices_ptr + indices_start + thread_id, mask=thread_id < K, other=-1)

    for k in range(K):
        idx = sparse_indices[k]
        if idx < 0 or idx >= S:
            continue  # Skip invalid indices

        # Propagate rewards backward from sparse indices
        for t in range(idx - 1, -1, -1):  # Reverse propagation
            out_t = base_idx + t
            out_next = base_idx + (t + 1)

            r_next = tl.load(rewards_ptr + out_next)
            r_t = tl.load(rewards_ptr + out_t)

            r_t += discount * r_next  # Discounted accumulation
            tl.store(out_ptr + out_t, r_t)

def sparse_reward_propagation_triton(rewards, indices, discount=0.99):
    """
    Calls the Triton kernel for sparse reward propagation.
    
    Args:
        rewards (torch.Tensor): Shape (B, S), batch of rewards.
        indices (torch.Tensor): Shape (B, K), sparse reward indices.
        discount (float): Discount factor.

    Returns:
        torch.Tensor: Propagated rewards of shape (B, S).
    """
    B, S = rewards.shape
    K = indices.shape[1]  # Number of sparse elements per batch

    # Allocate output tensor
    out = rewards.clone()

    # Grid and block configuration
    BLOCK_SIZE = 32  # Warp size for better performance
    grid = (B,)  # One block per batch

    # Launch the Triton kernel
    sparse_reward_propagation_kernel[grid](
        rewards, indices, out,
        B, S, K, discount,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
