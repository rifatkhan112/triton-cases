import torch

def sparse_reward_propagation_naive(rewards, indices, discount=0.99):
    """
    Naive implementation of sparse reward propagation.
    
    Args:
        rewards (torch.Tensor): Shape (B, S), where B=batch size, S=sequence length.
        indices (torch.Tensor): Shape (B, K), stores sparse reward indices for each batch.
        discount (float): Discount factor for reward propagation.

    Returns:
        torch.Tensor: Propagated reward values of shape (B, S).
    """
    B, S = rewards.shape
    out = rewards.clone()

    # Iterate backward in time for each sequence in batch
    for b in range(B):
        for k in range(indices.shape[1]):  # Iterate over sparse reward indices
            idx = indices[b, k].item()
            if idx < 0 or idx >= S:
                continue  # Skip invalid indices
            
            # Propagate reward backward in sequence
            for t in reversed(range(idx)):
                out[b, t] += discount * out[b, t + 1]

    return out
