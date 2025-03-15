import torch
import numpy as np
import time
from typing import Callable, List, Tuple, Union, Optional

from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def test_forward():
    """Validate forward pass with various edge cases"""
    tests = [
        # (rewards, dones, discount)
        # Test 0: Basic reward at beginning with terminal at end
        (
            [[1, 0, 0, 0], [0, 0, 0, 1]],
            [[0, 0, 0, 1], [0, 0, 0, 1]],
            0.9
        ),
        # Test 1: Reward at second position with terminal after it
        (
            [[0, 2, 0, 0], [1, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 0, 0, 1]],
            0.8
        ),
        # Test 2: Empty rewards
        (
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            0.95
        ),
        # Test 3: All terminal states
        (
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            0.9
        ),
        # Test 4: Multiple rewards without terminals
        (
            [[1, 0, 2, 0], [0, 3, 0, 4]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            0.9
        )
    ]
    
    for i, (rewards, dones, discount) in enumerate(tests):
        rewards = torch.tensor(rewards, device="cuda", dtype=torch.float32)
        dones = torch.tensor(dones, device="cuda", dtype=torch.float32)
        
        # Run both implementations
        naive_out = sparse_reward_propagation_naive(rewards, discount, dones)
        triton_out = sparse_reward_propagation_triton(rewards, discount, dones)
        
        # Print outputs for debugging
        print(f"Test {i}:")
        print(f"Inputs: rewards={rewards.cpu().numpy().tolist()}, dones={dones.cpu().numpy().tolist()}")
        print(f"Naive output: {naive_out.cpu().numpy().tolist()}")
        print(f"Triton output: {triton_out.cpu().numpy().tolist()}")
        print("-------")
        
        # Check if implementations match each other
        assert torch.allclose(naive_out, triton_out, atol=1e-4), f"Test {i}: Implementations don't match"


def test_gradients():
    """Verify gradient calculations"""
    torch.manual_seed(42)
    B, S = 2, 4
    rewards = torch.randn(B, S, device="cuda", requires_grad=True)
    rewards_triton = rewards.clone().requires_grad_(True)
    dones = torch.bernoulli(torch.full((B, S), 0.2, device="cuda"))
    
    # Test naive implementation
    out_naive = sparse_reward_propagation_naive(rewards, 0.9, dones)
    out_naive.sum().backward()
    assert rewards.grad is not None, "Naive: Gradients not calculated"
    assert not torch.isnan(rewards.grad).any(), "Naive: NaN in gradients"
    
    # Test triton implementation
    out_triton = sparse_reward_propagation_triton(rewards_triton, 0.9, dones)
    out_triton.sum().backward()
    assert rewards_triton.grad is not None, "Triton: Gradients not calculated"
    assert not torch.isnan(rewards_triton.grad).any(), "Triton: NaN in gradients"
    
    # Compare gradients
    assert torch.allclose(rewards.grad, rewards_triton.grad, atol=1e-4), "Gradient mismatch between implementations"


def test_none_dones():
    """Test with None dones parameter"""
    B, S = 2, 4
    rewards = torch.tensor([[1.0, 0.0, 0.0, 2.0], [0.0, 3.0, 0.0, 0.0]], device="cuda")
    
    # Test naive implementation
    naive_with_dones = sparse_reward_propagation_naive(
        rewards, 0.9, torch.zeros((B, S), device="cuda")
    )
    naive_without_dones = sparse_reward_propagation_naive(rewards, 0.9, None)
    assert torch.allclose(naive_with_dones, naive_without_dones), "Naive: None dones handling failed"
    
    # Test triton implementation
    triton_with_dones = sparse_reward_propagation_triton(
        rewards, 0.9, torch.zeros((B, S), device="cuda")
    )
    triton_without_dones = sparse_reward_propagation_triton(rewards, 0.9, None)
    assert torch.allclose(triton_with_dones, triton_without_dones), "Triton: None dones handling failed"
    
    # Compare implementations
    assert torch.allclose(naive_with_dones, triton_with_dones), "Implementations mismatch with zero dones"


def test_performance(verbose=True):
    """Benchmark performance of implementations with different sparsity levels"""
    torch.manual_seed(42)
    
    test_configs = [
        # (batch_size, seq_length, sparsity, discount, done_prob, num_runs)
        (32, 128, 0.99, 0.99, 0.01, 3),    # Very sparse, realistic RL scenario
        (32, 128, 0.5, 0.99, 0.1, 3),      # Medium sparsity
        (32, 128, 0.0, 0.99, 0.05, 3),     # Dense (no sparsity)
    ]
    
    results = []
    
    for B, S, sparsity, discount, done_prob, num_runs in test_configs:
        # Create input tensors
        raw_rewards = torch.randn(B, S, device="cuda")
        if sparsity > 0:
            # Create sparse rewards by zeroing out some elements
            mask = torch.rand(B, S, device="cuda") < sparsity
            raw_rewards = raw_rewards * (~mask)
        
        dones = torch.bernoulli(torch.full((B, S), done_prob, device="cuda"))
        
        # Warmup
        _ = sparse_reward_propagation_naive(raw_rewards, discount, dones)
        _ = sparse_reward_propagation_triton(raw_rewards, discount, dones)
        torch.cuda.synchronize()
        
        # Time naive implementation
        naive_times = []
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = sparse_reward_propagation_naive(raw_rewards, discount, dones)
            end.record()
            torch.cuda.synchronize()
            naive_times.append(start.elapsed_time(end))
        
        # Time triton implementation
        triton_times = []
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = sparse_reward_propagation_triton(raw_rewards, discount, dones)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))
        
        naive_avg = sum(naive_times) / len(naive_times)
        triton_avg = sum(triton_times) / len(triton_times)
        speedup = naive_avg / triton_avg if triton_avg > 0 else 0
        
        results.append({
            'B': B,
            'S': S,
            'sparsity': sparsity,
            'naive_ms': naive_avg,
            'triton_ms': triton_avg,
            'speedup': speedup
        })
        
        if verbose:
            non_zero = int(B * S * (1 - sparsity))
            print(f"B={B}, S={S}, Non-zeros={non_zero}/{B*S} ({(1-sparsity)*100:.1f}%):")
            print(f"  Naive: {naive_avg:.3f} ms")
            print(f"  Triton: {triton_avg:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    print("Running forward tests...")
    test_forward()
    
    print("\nRunning gradient tests...")
    test_gradients()
    
    print("\nTesting None dones handling...")
    test_none_dones()
    
    print("\nRunning performance benchmarks...")
    results = test_performance(verbose=True)
    
    print("\nAll tests completed!")
