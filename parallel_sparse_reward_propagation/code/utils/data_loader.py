import torch

def load_sparse_transitions(batch_size=10):
    """Generate a batch of sparse RL state transitions."""
    states = torch.arange(batch_size)
    rewards = torch.zeros(batch_size, dtype=torch.float32)
    rewards[2] = 1.0
    rewards[6] = 2.0
    return states, rewards