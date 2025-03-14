import os
import torch
import matplotlib.pyplot as plt  # For displaying the generated plot
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
        plot_name="sparse_reward_propagation_performance",  # Plot name (PNG file)
        args={},
    )
)
def benchmark(batch_size, provider):
    """
    Benchmarking function to compare sparse reward propagation using different implementations.
    - Handles both forward and backward pass tests.
    - Ensures numerical stability in sparse settings.
    - Evaluates performance under different sparsity levels (5% and 0.1%).
    - Automatically generates a table and a PNG plot using Triton's perf_report.
    """
    device = 'cuda'
    dtype = torch.float32
    sequence_length = 4096  # Large sequence length for benchmarking
    requires_grad = True

    # Create input tensors for two different sparsity levels
    rewards_standard = torch.zeros((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=requires_grad)
    rewards_extreme = rewards_standard.clone()

    # Introduce standard sparsity (5% non-zero rewards)
    mask_standard = torch.rand_like(rewards_standard) < 0.05
    rewards_standard[mask_standard] = torch.randn_like(rewards_standard[mask_standard]).detach()

    # Extreme sparsity test case (0.1% non-zero rewards)
    mask_extreme = torch.rand_like(rewards_extreme) < 0.001
    rewards_extreme[mask_extreme] = torch.randn_like(rewards_extreme[mask_extreme]).detach()

    do = torch.ones_like(rewards_standard, dtype=dtype)  # Gradient tensor for backward pass
    quantiles = [0.5, 0.2, 0.8]  # Median, 20th, 80th percentile

    # Depending on the provider, run bench for standard or extreme (pick one for each call)
    # If you want separate calls for standard vs. extreme, you'd define them here.
    results_standard = (0, 0, 0)
    results_extreme = (0, 0, 0)

    # Run benchmark based on provider (only standard for the sake of perf_report)
    if provider == 'torch_fwd':
        # Standard
        median_std, pc_std = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards_standard.cpu(), 0.99).cuda(), quantiles=quantiles)
        low_std, high_std = pc_std

        # Extreme
        median_ext, pc_ext = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards_extreme.cpu(), 0.99).cuda(), quantiles=quantiles)
        low_ext, high_ext = pc_ext

        results_standard = (median_std, low_std, high_std)
        results_extreme = (median_ext, low_ext, high_ext)

    elif provider == 'triton_fwd':
        # Standard
        median_std, pc_std = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards_standard, 0.99), quantiles=quantiles)
        low_std, high_std = pc_std

        # Extreme
        median_ext, pc_ext = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards_extreme, 0.99), quantiles=quantiles)
        low_ext, high_ext = pc_ext

        results_standard = (median_std, low_std, high_std)
        results_extreme = (median_ext, low_ext, high_ext)

    elif provider == 'torch_bwd':
        # Standard
        median_std, pc_std = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards_standard.cpu(), 0.99).cuda().backward(do), quantiles=quantiles)
        low_std, high_std = pc_std

        # Extreme
        median_ext, pc_ext = triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards_extreme.cpu(), 0.99).cuda().backward(do), quantiles=quantiles)
        low_ext, high_ext = pc_ext

        results_standard = (median_std, low_std, high_std)
        results_extreme = (median_ext, low_ext, high_ext)

    elif provider == 'triton_bwd':
        # Standard
        median_std, pc_std = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards_standard, 0.99).backward(do), quantiles=quantiles)
        low_std, high_std = pc_std

        # Extreme
        median_ext, pc_ext = triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards_extreme, 0.99).backward(do), quantiles=quantiles)
        low_ext, high_ext = pc_ext

        results_standard = (median_std, low_std, high_std)
        results_extreme = (median_ext, low_ext, high_ext)

    # Print a summary for standard and extreme
    print(f"\n[Provider: {provider}, batch_size={batch_size}]")
    ms_std, low_std, high_std = results_standard
    ms_ext, low_ext, high_ext = results_extreme

    print("5% Sparsity (Standard):")
    print(f"  Median: {ms_std:.5f} ms | 20th: {low_std:.5f} ms | 80th: {high_std:.5f} ms")

    print("0.1% Sparsity (Extreme):")
    print(f"  Median: {ms_ext:.5f} ms | 20th: {low_ext:.5f} ms | 80th: {high_ext:.5f} ms")

    # Return standard results to comply with perf_report's requirement
    return results_standard  # or results_extreme if you prefer


if __name__ == '__main__':
    # 1) Run the benchmark (prints table + saves plot automatically)
    benchmark.run(print_data=True)

    # 2) Attempt to show the generated plot
    plot_filename = "sparse_reward_propagation_performance.png"
    if os.path.exists(plot_filename):
        print(f"\n📊 Plot saved as '{plot_filename}'. Displaying now...")
        img = plt.imread(plot_filename)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Sparse Reward Propagation Performance")
        plt.show()
    else:
        print("⚠️ No plot file found. Check if the benchmark ran successfully.")
