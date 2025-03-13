import torch
from torch.utils.data import Dataset, DataLoader

class SparseRewardDataset(Dataset):
    """
    A PyTorch Dataset for generating sparse reinforcement learning data.
    """

    def __init__(self, batch_size, sequence_length, sparsity=0.05, device="cuda", seed=42):
        """
        Args:
            batch_size (int): Number of sequences in a batch.
            sequence_length (int): Number of timesteps per sequence.
            sparsity (float): Fraction of non-zero rewards (e.g., 0.05 means 5% are non-zero).
            device (str): "cuda" or "cpu".
            seed (int): Random seed for reproducibility.
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.sparsity = sparsity
        self.device = device

        torch.manual_seed(seed)

        # Generate state transitions (random numbers)
        self.states = torch.randn((batch_size, sequence_length), dtype=torch.float32, device=device)

        # Generate sparse rewards
        self.rewards = torch.zeros((batch_size, sequence_length), dtype=torch.float32, device=device)
        mask = torch.rand_like(self.rewards) < self.sparsity
        self.rewards[mask] = torch.randn_like(self.rewards[mask])

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return self.states[idx], self.rewards[idx]

def get_data_loader(batch_size=4096, sequence_length=100, sparsity=0.05, device="cuda", num_workers=2):
    """
    Returns a DataLoader for the SparseRewardDataset.

    Args:
        batch_size (int): Number of sequences per batch.
        sequence_length (int): Number of timesteps per sequence.
        sparsity (float): Fraction of non-zero rewards.
        device (str): "cuda" or "cpu".
        num_workers (int): Number of workers for data loading.
    
    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = SparseRewardDataset(batch_size, sequence_length, sparsity, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))

# ✅ Example Usage
if __name__ == "__main__":
    data_loader = get_data_loader(batch_size=2048, sequence_length=100, sparsity=0.05, device="cuda")

    for states, rewards in data_loader:
        print(f"🟢 Batch Shape: {states.shape}, {rewards.shape}")
        print(f"🔹 Non-Zero Rewards: {(rewards != 0).sum().item()} / {rewards.numel()} (Sparsity: {100 * (rewards != 0).sum().item() / rewards.numel():.2f}%)")
        break  # Test one batch only