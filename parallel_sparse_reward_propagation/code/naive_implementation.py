import torch

def sparse_reward_propagation_naive(rewards, discount=0.99):
    """
    B, S = rewards.shape
    For t in reversed(range(S-1)):
        out[:, t] += discount * out[:, t+1]
    """
    out = rewards.clone()
    B, S = out.shape
    for t in reversed(range(S - 1)):
        out[:, t] += discount * out[:, t + 1]
    return out
