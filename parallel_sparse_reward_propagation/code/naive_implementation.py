import torch

def sparse_reward_propagation_naive(rewards, discount=0.99):
    """
    Naive sequential accumulation:
    out[:, t] += discount * out[:, t+1]
    """
    B, S = rewards.shape
    out = rewards.clone()

    for t in reversed(range(S - 1)):
        out[:, t] += discount * out[:, t + 1]

    return out
