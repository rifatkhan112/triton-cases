import torch

def sparse_reward_propagation_naive(states, rewards, gamma=0.99):
    """Naive CPU implementation of sparse reward propagation."""
    propagated_rewards = torch.zeros_like(rewards)
    for i in range(len(states) - 2, -1, -1):
        if rewards[i] != 0:
            propagated_rewards[i] = rewards[i]
        else:
            propagated_rewards[i] = gamma * propagated_rewards[i + 1]
    return propagated_rewards

if __name__ == "__main__":
    states = torch.arange(10)
    rewards = torch.tensor([0, 0, 1, 0, 0, 0, 2, 0, 0, 0], dtype=torch.float32)
    result = sparse_reward_propagation_naive(states, rewards)
    print("Naive Propagated Rewards:", result)