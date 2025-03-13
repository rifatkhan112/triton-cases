import torch

def sparse_reward_propagation_naive(rewards, discount=0.99):
    """
    Naive implementation of backward pass for RL reward propagation.
    (B, S) => (B, S)
    """
    B, S = rewards.shape
    out = rewards.clone()
    # Simple backward pass: out[:, t] += discount * out[:, t+1]
    for t in reversed(range(S - 1)):
        out[:, t] = out[:, t] + discount * out[:, t + 1]
    return out
