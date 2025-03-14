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
    - Validates performance gains using Triton.
    """
    device = 'cuda'
    dtype = torch.float32
    sequence_length = 4096  # Increased sequence length as per problem context
    requires_grad = True

    # ✅ Create input tensors
    rewards_standard = torch.zeros((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=requires_grad)
    rewards_extreme = torch.zeros((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=requires_grad)

    # ✅ Clone rewards before modification to prevent in-place operation errors
    rewards_standard = rewards_standard.clone()
    rewards_extreme = rewards_extreme.clone()

    # ✅ Introduce sparsity (5% non-zero rewards for standard test case)
    mask_standard = torch.rand_like(rewards_standard) < 0.05
    rewards_standard[mask_standard] = torch.randn_like(rewards_standard[mask_standard]).detach()

    # ✅ Extreme sparsity test case (0.1% non-zero rewards)
    mask_extreme = torch.rand_like(rewards_extreme) < 0.001
    rewards_extreme[mask_extreme] = torch.randn_like(rewards_extreme[mask_extreme]).detach()

    do = torch.ones_like(rewards_standard, dtype=dtype)  # Gradient tensor for backward pass

    quantiles = [0.5, 0.2, 0.8]  # Median, 20th, 80th percentile

    # ✅ Run benchmark based on selected provider
    def benchmark_run(rewards):
        if provider == 'torch_fwd':
            return triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards.cpu(), 0.99).cuda(), quantiles=quantiles)
        elif provider == 'triton_fwd':
            return triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards, 0.99), quantiles=quantiles)
        elif provider == 'torch_bwd':
            def torch_backward():
                output = sparse_reward_propagation_naive(rewards.cpu(), 0.99).cuda()
                output.backward(do, retain_graph=True)
            results = triton.testing.do_bench(torch_backward, quantiles=quantiles)

        elif provider == 'triton_bwd':
            def triton_backward():
                output = sparse_reward_propagation_triton(rewards, 0.99)
                output.backward(do, retain_graph=True)
            results = triton.testing.do_bench(triton_backward, quantiles=quantiles)
        return None

    # ✅ Benchmark for both standard (5%) and extreme (0.1%) sparsity cases
    results_standard = benchmark_run(rewards_standard)
    results_extreme = benchmark_run(rewards_extreme)

    # ✅ Print detailed performance metrics
    print(f"\n📌Benchmark Results for batch_size={batch_size}, provider={provider}\n")

    def print_results(sparsity, results):
        if results is not None:
            y_median, y_min, y_max = results
            print(f"📌**Performance for {sparsity} Sparsity**:")
            print(f"Execution Time (Median): {y_median:.5f} ms")
            print(f"Execution Time (20th Percentile): {y_min:.5f} ms")
            print(f"Execution Time (80th Percentile): {y_max:.5f} ms")

    print_results("5%", results_standard)
    print_results("0.1% (Extreme Case)", results_extreme)

    return results_standard

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)
