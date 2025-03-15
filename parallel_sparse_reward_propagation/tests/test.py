import torch

from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def test_implementations_match():
    """Test that naive and triton implementations produce the same results"""
    print("Testing with simple cases...")
    
    # Simple test cases
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], device="cuda")
    dones = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], device="cuda")
    
    naive_out = sparse_reward_propagation_naive(rewards, 0.9, dones)
    triton_out = sparse_reward_propagation_triton(rewards, 0.9, dones)
    
    print("Naive output:", naive_out)
    print("Triton output:", triton_out)
    
    match = torch.allclose(naive_out, triton_out, atol=1e-5)
    print(f"Simple test: {'PASSED' if match else 'FAILED'}")
    
    print("\nTesting with random data...")
    torch.manual_seed(42)
    
    # Random test
    B, S = 4, 8
    rewards = torch.randn(B, S, device="cuda")
    dones = torch.bernoulli(torch.full((B, S), 0.2, device="cuda"))
    
    naive_out = sparse_reward_propagation_naive(rewards, 0.95, dones)
    triton_out = sparse_reward_propagation_triton(rewards, 0.95, dones)
    
    match = torch.allclose(naive_out, triton_out, atol=1e-5)
    if not match:
        print("Random test: FAILED")
        diff = (naive_out - triton_out).abs()
        max_diff = diff.max().item()
        print(f"Max difference: {max_diff}")
        
        # Print indices where the difference is largest
        max_idx = diff.argmax().item()
        b_idx = max_idx // S
        s_idx = max_idx % S
        print(f"Largest difference at [b={b_idx}, s={s_idx}]:")
        print(f"  Naive: {naive_out[b_idx, s_idx].item()}")
        print(f"  Triton: {triton_out[b_idx, s_idx].item()}")
        return False
    else:
        print("Random test: PASSED")
    
    print("\nAll tests passed!")
    return True


def benchmark_performance():
    """Benchmark performance of both implementations"""
    print("\nBenchmarking performance...")
    torch.manual_seed(42)
    
    # Test configurations
    configs = [
        # (batch_size, seq_length, sparsity, done_prob, discount)
        (32, 128, 0.99, 0.01, 0.99),  # Very sparse
        (32, 128, 0.5, 0.1, 0.95),    # Medium sparsity
        (32, 128, 0.0, 0.05, 0.9),    # Dense
    ]
    
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
        
        non_zero = int(B * S * (1 - sparsity))
        print(f"B={B}, S={S}, Non-zeros={non_zero}/{B*S} ({(1-sparsity)*100:.1f}%):")
        print(f"  Naive: {naive_time:.3f} ms")
        print(f"  Triton: {triton_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    if test_implementations_match():
        benchmark_performance()
    else:
        print("Tests failed! Skipping benchmarks.")
