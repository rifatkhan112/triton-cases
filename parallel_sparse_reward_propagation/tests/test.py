import torch
import time
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def exact_compare(naive_fn, triton_fn, B=4, S=4096, device="cuda"):
    """Direct comparison preserving your original test logic with enhancements"""
    # Initialize identical inputs for both implementations
    rewards = torch.randn(B, S, device=device, requires_grad=True)
    dones = torch.bernoulli(torch.full((B, S), 0.1, device=device))
    
    # Clone for gradient checks
    rewards_naive = rewards.clone().detach().requires_grad_()
    rewards_triton = rewards.clone().detach().requires_grad_()

    # Forward passes
    ref_out = naive_fn(rewards_naive, dones=dones)
    tri_out = triton_fn(rewards_triton, dones=dones)
    
    # 1. Strict numerical comparison (matches your original tolerance)
    close = torch.allclose(ref_out, tri_out, atol=1e-5, rtol=1e-5)
    max_diff = (ref_out - tri_out).abs().max().item()
    print(f"Forward match: {close} | Max difference: {max_diff:.2e}")

    # 2. Gradient comparison with your original methodology
    grad_out = torch.ones_like(ref_out)
    
    ref_out.backward(grad_out)
    tri_out.backward(grad_out)
    
    grad_close = torch.allclose(rewards_naive.grad, rewards_triton.grad, atol=1e-5)
    grad_max_diff = (rewards_naive.grad - rewards_triton.grad).abs().max().item()
    print(f"Gradient match: {grad_close} | Max grad difference: {grad_max_diff:.2e}")

    return close and grad_close

def benchmark_original_style():
    """Performance comparison matching your original timing approach"""
    B, S = 256, 4096
    discount = 0.99
    reps = 100
    
    # Generate test data
    rewards = torch.randn(B, S, device="cuda")
    dones = torch.bernoulli(torch.full((B, S), 0.1, device="cuda"))
    
    # Naive timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps):
        _ = sparse_reward_propagation_naive(rewards, discount, dones)
    torch.cuda.synchronize()
    naive_time = (time.time() - start) / reps
    
    # Triton timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps):
        _ = sparse_reward_propagation_triton(rewards, discount, dones)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / reps
    
    print(f"\nNaive avg: {naive_time*1000:.2f}ms")
    print(f"Triton avg: {triton_time*1000:.2f}ms")
    print(f"Speedup: {naive_time/triton_time:.1f}x")

if __name__ == "__main__":
    # Preserve your original test flow
    print("=== Running original-style validation ===")
    success = exact_compare(
        sparse_reward_propagation_naive,
        sparse_reward_propagation_triton,
        B=4,
        S=4096
    )
    
    print("\n=== Benchmarking ===")
    benchmark_original_style()
    
    if success:
        print("\nTest passed: Implementations match!")
    else:
        print("\nTest failed: Implementations diverge!")
