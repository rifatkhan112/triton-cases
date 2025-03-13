import torch
import triton
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['Batch Size'],
        x_vals=[1024 * 2 ** i for i in range(0, 6)],  # Test different batch sizes
        line_arg='provider',
        line_vals=['torch_fwd', 'triton_fwd', 'torch_bwd', 'triton_bwd'],
        line_names=['Torch Forward', 'Triton Forward', 'Torch Backward', 'Triton Backward'],
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",
        plot_name="Sparse Reward Propagation Performance",
        args={},
    )
)
def benchmark(batch_size, provider):
    device = 'cuda'
    dtype = torch.float32
    sequence_length = 100  # Fixed sequence length
    requires_grad = True

    # Create input tensors
    states = torch.randn((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=requires_grad)
    rewards = torch.zeros((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=requires_grad)
    
    # Introduce sparsity (5% non-zero rewards)
    mask = torch.rand_like(rewards) < 0.05
    rewards[mask] = torch.randn_like(rewards[mask])

    do = torch.ones_like(rewards, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]  # Median, 20th, 80th percentile
    results = 0, 0, 0
    
    # Run benchmark based on selected provider
    if provider == 'torch_fwd':
        results = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(states.cpu(), rewards.cpu()).cuda(), quantiles=quantiles)
    elif provider == 'triton_fwd':
        results = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(states, rewards), quantiles=quantiles)
    elif provider == 'torch_bwd':
        results = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(states.cpu(), rewards.cpu()).cuda().backward(do), quantiles=quantiles)
    elif provider == 'triton_bwd':
        results = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(states, rewards).backward(do), quantiles=quantiles)
    
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
