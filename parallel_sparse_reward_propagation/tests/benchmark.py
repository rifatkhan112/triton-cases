import torch
import triton
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size'],  # Batch sizes for benchmarking
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
    """
    Benchmarking function to compare sparse reward propagation using different implementations.
    - Handles both forward and backward pass tests.
    - Ensures numerical stability in sparse settings.
    - Evaluates performance under different sparsity levels (5% and 0.1%).
    """
    device = 'cuda'
    dtype = torch.float32
    sequence_length = 4096  # Large sequence length for benchmarking
    requires_grad = True

    # ✅ Create input tensors
    rewards = torch.zeros((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=requires_grad)

    # ✅ Introduce standard sparsity (5% non-zero rewards)
    standard_mask = torch.rand_like(rewards) < 0.05  # 5% sparsity
    rewards_standard = rewards.clone()
    rewards_standard[standard_mask] = torch.randn_like(rewards_standard[standard_mask]).detach()

    # ✅ Introduce extreme sparsity (0.1% non-zero rewards)
    extreme_mask = torch.rand_like(rewards) < 0.001  # 0.1% sparsity
    rewards_extreme = rewards.clone()
    rewards_extreme[extreme_mask] = torch.randn_like(rewards_extreme[extreme_mask]).detach()

    do = torch.ones_like(rewards, dtype=dtype)  # Gradient tensor for backward pass

    quantiles = [0.5, 0.2, 0.8]  # Median, 20th, 80th percentile
    results_standard = 0, 0, 0
    results_extreme = 0, 0, 0

    # ✅ Run benchmark for both sparsity levels
    if provider == 'torch_fwd':
        results_standard = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards_standard.cpu(), 0.99).cuda(), quantiles=quantiles)
        results_extreme = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards_extreme.cpu(), 0.99).cuda(), quantiles=quantiles)
    elif provider == 'triton_fwd':
        results_standard = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards_standard, 0.99), quantiles=quantiles)
        results_extreme = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards_extreme, 0.99), quantiles=quantiles)
    elif provider == 'torch_bwd':
        results_standard = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards_standard.cpu(), 0.99).cuda().backward(do), quantiles=quantiles)
        results_extreme = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards_extreme.cpu(), 0.99).cuda().backward(do), quantiles=quantiles)
    elif provider == 'triton_bwd':
        results_standard = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards_standard, 0.99).backward(do), quantiles=quantiles)
        results_extreme = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards_extreme, 0.99).backward(do), quantiles=quantiles)

    # ✅ Compute numerical stability metrics separately for both sparsity levels
    naive_standard = sparse_reward_propagation_naive(rewards_standard.cpu(), 0.99).cuda()
    triton_standard = sparse_reward_propagation_triton(rewards_standard, 0.99)
    mae_standard = torch.mean(torch.abs(naive_standard - triton_standard)).item()
    mad_standard = torch.max(torch.abs(naive_standard - triton_standard)).item()
    percent_diff_standard = (torch.abs(naive_standard - triton_standard) / (torch.abs(naive_standard) + 1e-8)).mean().item()

    naive_extreme = sparse_reward_propagation_naive(rewards_extreme.cpu(), 0.99).cuda()
    triton_extreme = sparse_reward_propagation_triton(rewards_extreme, 0.99)
    mae_extreme = torch.mean(torch.abs(naive_extreme - triton_extreme)).item()
    mad_extreme = torch.max(torch.abs(naive_extreme - triton_extreme)).item()
    percent_diff_extreme = (torch.abs(naive_extreme - triton_extreme) / (torch.abs(naive_extreme) + 1e-8)).mean().item()

    # ✅ Print Execution Results for both sparsity levels
    print(f"\n🚀 Benchmark Results for batch_size={batch_size}, provider={provider}")
    
    print("\n📌 **Performance for 5% Sparsity**:")
    print(f"Execution Time (Median): {results_standard[0]:.5f} ms")
    print(f"Execution Time (20th Percentile): {results_standard[1]:.5f} ms")
    print(f"Execution Time (80th Percentile): {results_standard[2]:.5f} ms")
    print(f"Numerical Stability Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae_standard:.5e}")
    print(f"  - Maximum Absolute Difference (MAD): {mad_standard:.5e}")
    print(f"  - Mean Percentage Difference: {percent_diff_standard * 100:.5f}%\n")

    print("\n📌 **Performance for 0.1% Sparsity (Extreme Case)**:")
    print(f"Execution Time (Median): {results_extreme[0]:.5f} ms")
    print(f"Execution Time (20th Percentile): {results_extreme[1]:.5f} ms")
    print(f"Execution Time (80th Percentile): {results_extreme[2]:.5f} ms")
    print(f"Numerical Stability Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae_extreme:.5e}")
    print(f"  - Maximum Absolute Difference (MAD): {mad_extreme:.5e}")
    print(f"  - Mean Percentage Difference: {percent_diff_extreme * 100:.5f}%\n")

    return results_standard, results_extreme

if __name__ == '__main__':
    benchmark.run(print_data=True)
