import torch
import triton
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1024 * 2 ** i for i in range(0, 6)],
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
    sequence_length = 4096

    # ✅ GPU Check
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA device not available. Ensure you're running on GPU.")
    print("🚀GPU Check Passed. Device:", torch.cuda.get_device_name(0))

    # ✅ Standard sparsity case (5%)
    rewards_standard = torch.zeros((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=True)
    mask_standard = torch.rand_like(rewards_standard) < 0.05
    rewards_standard = rewards_standard.clone()
    rewards_standard[mask_standard] = torch.randn_like(rewards_standard[mask_standard]).detach()

    # ✅ Extreme sparsity case (0.1%)
    rewards_extreme = torch.zeros((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=True)
    mask_extreme = torch.rand_like(rewards_extreme) < 0.001
    rewards_extreme = rewards_extreme.clone()
    rewards_extreme[mask_extreme] = torch.randn_like(rewards_extreme[mask_extreme]).detach()

    do = torch.ones_like(rewards_standard, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]

    def benchmark_run(rewards):
        if provider == 'torch_fwd':
            return triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards.cpu(), 0.99).cuda(), quantiles=quantiles)
        elif provider == 'triton_fwd':
            return triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards, 0.99), quantiles=quantiles)
        elif provider == 'torch_bwd':
            return triton.testing.do_bench(lambda: sparse_reward_propagation_naive(rewards.cpu(), 0.99).cuda().backward(do, retain_graph=True), quantiles=quantiles)
        elif provider == 'triton_bwd':
            return triton.testing.do_bench(lambda: sparse_reward_propagation_triton(rewards, 0.99).backward(do, retain_graph=True), quantiles=quantiles)

    # ✅ Run benchmarks
    print("\n📌 Benchmark Results for batch_size={}, provider={}".format(batch_size, provider))

    results_standard = benchmark_run(rewards_standard)
    print("\n📌 **Performance for 5% Sparsity**:")
    print(f"Execution Time (Median): {results_standard[0]:.5f} ms")
    print(f"Execution Time (20th Percentile): {results_standard[1]:.5f} ms")
    print(f"Execution Time (80th Percentile): {results_standard[2]:.5f} ms")

    results_extreme = benchmark_run(rewards_extreme)
    print("\n📌 **Performance for 0.1% (Extreme Case) Sparsity**:")
    print(f"Execution Time (Median): {results_extreme[0]:.5f} ms")
    print(f"Execution Time (20th Percentile): {results_extreme[1]:.5f} ms")
    print(f"Execution Time (80th Percentile): {results_extreme[2]:.5f} ms")

    return results_standard

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)
