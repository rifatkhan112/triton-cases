import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def test_forward():
    """Validate forward pass with edge cases"""
    tests = [
        # (rewards, dones, expected)
        (
            [[1, 0, 0, 0], [0, 0, 0, 1]],
            [[0, 0, 0, 1], [0, 0, 0, 1]],
            0.9,
            [
                [1*0.9**3, 1*0.9**2, 1*0.9, 1],
                [0, 0, 0, 1]
            ]
        ),
        (
            [[0, 2, 0, 0], [1, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 0, 0, 1]],
            0.8,
            [
                [0, 2, 0, 0],
                [1 + 0.8*0 + 0.8**2*0 + 0.8**3*0, 0, 0, 0]
            ]
        )
    ]
    
    for rewards, dones, discount, expected in tests:
        rewards = torch.tensor(rewards, device="cuda", dtype=torch.float32)
        dones = torch.tensor(dones, device="cuda", dtype=torch.float32)
        
        naive_out = sparse_reward_propagation_naive(rewards, discount, dones)
        triton_out = sparse_reward_propagation_triton(rewards, discount, dones)
        expected_tensor = torch.tensor(expected, device="cuda")
        
        assert torch.allclose(naive_out, expected_tensor, atol=1e-4), "Naive mismatch"
        assert torch.allclose(triton_out, expected_tensor, atol=1e-4), "Triton mismatch"

def test_gradients():
    """Verify gradient calculations"""
    torch.manual_seed(42)
    B, S = 2, 4
    rewards = torch.randn(B, S, device="cuda", requires_grad=True)
    dones = torch.bernoulli(torch.full((B, S), 0.2, device="cuda"))
    
    for impl in [sparse_reward_propagation_naive, sparse_reward_propagation_triton]:
        rewards.grad = None
        out = impl(rewards.clone(), 0.9, dones)
        out.sum().backward()
        assert rewards.grad is not None, "Gradients not calculated"
        assert not torch.isnan(rewards.grad).any(), "NaN in gradients"

if __name__ == "__main__":
    test_forward()
    test_gradients()
    print("All tests passed successfully!")
