import torch
import time

from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def test_implementations_match():
    """Test that naive and triton implementations produce the same results"""
    torch.manual_seed(42)
    
    test_cases = [
        # Simple test
        (torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], device="cuda"), 
         torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], device="cuda"), 
         0.9),
        
        # With terminal states
        (torch.tensor([[0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device="cuda"), 
         torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], device="cuda"), 
         0.8),
        
        # Random case
        (torch.randn(4, 8, device="cuda"), 
         torch.bernoulli(torch.full((4, 8), 0.2, device="cuda")), 
         0.95),
    ]
    
    for i, (rewards, dones, discount) in enumerate(test_cases):
        # Get outputs from both implementations
        naive_out = sparse_reward_propagation_naive(rewards, discount, dones)
        triton_out = sparse_reward_propagation_triton(rewards, discount, dones)
        
        # Check if outputs match
        if not torch.allclose(naive_out, triton_out, atol=1e-4):
            print(f"Test case {i} failed")
            print(f"Rewards: {rewards.cpu().numpy()}")
            print(f"Dones: {dones.cpu().numpy()}")
            print(f"Naive output: {naive_out.cpu().numpy()}")
            print(f"Triton output: {triton_out.cpu().numpy()}")
            print(f"Difference: {(naive_out - triton_out).abs().max().item()}")
            return False
        else:
            print(f"Test case {i} passed")
    
    return True


def benchmark_performance():
    """Benchmark performance of both implementations"""
    torch.manual_seed(42)
    
    # Test configurations
    configs = [
        # (batch_size, seq_length, sparsity, done_prob, discount)
        (32, 128, 0.99, 0.01, 0.99),  # Very sparse
        (32, 128, 0.5, 0.1, 0.95),    # Medium sparsity
        (32, 128, 0.0, 0.05, 0.9),    # Dense
    ]
    
    results = []
    
    for B, S, sparsity, done_prob, discount in configs:
        # Create input tensors
        rewards = torch.randn(B, S, device="cuda")
        if sparsity > 0:
            # Create sparse rewards by zeroing out elements
            mask = torch.rand(B, S, device="cuda") < sparsity
            rewards = rewards * (~mask)
        
        dones = torch.bernoulli(torch.full((B, S), done_prob, device="cuda"))
        
        # Warmup
        _ = sparse_reward_propagation_naive(rewards, discount, dones)
        _ = sparse_reward_propagation_triton(rewards, discount, dones)
        torch.cuda.synchronize()
        
        # Benchmark naive implementation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            _ = sparse_reward_propagation_naive(rewards, discount, dones)
        end.record()
        torch.cuda.synchronize()
        naive_time = start.elapsed_time(end) / 10
        
        # Benchmark Triton implementation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            _ = sparse_reward_propagation_triton(rewards, discount, dones)
        end.record()
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 10
        
        speedup = naive_time / triton_time if triton_time > 0 else 0
        
        print(f"B={B}, S={S}, Sparsity={sparsity*100:.1f}%:")
        print(f"  Naive: {naive_time:.3f} ms")
        print(f"  Triton: {triton_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            'B': B, 'S': S, 'sparsity': sparsity,
            'naive_ms': naive_time, 'triton_ms': triton_time,
            'speedup': speedup
        })
    
    return results


if __name__ == "__main__":
    print("Testing implementations match...")
    if test_implementations_match():
        print("All tests passed!")
        
        print("\nBenchmarking performance...")
        results = benchmark_performance()
        print("\nBenchmark complete!")
    else:
        print("Tests failed!")
