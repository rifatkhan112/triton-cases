import torch

def sparse_reward_propagation_naive(rewards, sparse_indices, discount=0.99):
    B, S = rewards.shape
    _, K = sparse_indices.shape
    out = rewards.clone()

    for b in range(B):
        for k in range(K):
            idx = sparse_indices[b, k]
            if idx >= 0 and idx < S - 1:
                out[b, idx] += discount * out[b, idx + 1]
    
    return out
