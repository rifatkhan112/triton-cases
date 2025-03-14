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

    rewards_standard = torch.zeros((batch_size, sequence_length), dtype=dtype, device=device, requires_grad=True)
    rewards_extreme = rewards_standard.clone()

    mask_standard = torch.rand_like(rewards_standard) < 0.05
    rewards_standard = rewards_standard.clone()
    rewards_standard[mask_standard] = torch.randn_like(rewards_standard[mask_standard]).detach()

    mask_extreme = torch.rand_like(rewards_standard) < 0.001
    rewards_extreme = rewards_standard.clone()
    rewards_extreme[mask_extreme] = torch.randn_like(rewards_extreme[mask_extreme]).detach()

    do = torch.ones_like(rewards_standard, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    results_standard = results_extreme = (0, 0, 0)

    if provider == 'torch_fwd':
        results_standard = triton.testing.do_bench(
            lambda: sparse_reward_propagation_naive(rewards_standard.cpu(), 0.99).cuda(), quantiles=quantiles)
        results_extreme = triton.testing.do_bench(
            lambda: sparse_reward_propagation_naive(rewards_extreme.cpu(), 0.99).cuda(), quantiles=quantiles)

    elif provider == 'triton_fwd':
        results_standard = triton.testing.do_bench(
            lambda: sparse_reward_propagation_triton(rewards_standard, 0.99), quantiles=quantiles)
        results_extreme = triton.testing.do_bench(
            lambda: sparse_reward_propagation_triton(rewards_extreme, 0.99), quantiles=quantiles)

    elif provider == 'torch_bwd':
        do = torch.ones_like(rewards_standard, dtype=dtype)
        results_standard = triton.testing.do_bench(
            lambda: sparse_reward_propagation_naive(rewards_standard.cpu(), 0.99).cuda().backward(do), quantiles=quantiles)
        results_extreme = triton.testing.do_bench(
            lambda: sparse_reward_propagation_naive(rewards_extreme.cpu(), 0.99).cuda().backward(do), quantiles=quantiles)

    elif provider == 'triton_bwd':
        do = torch.ones_like(rewards_standard, dtype=dtype)
        results_standard = triton.testing.do_bench(
            lambda: sparse_reward_propagation_triton(rewards_standard, 0.99).backward(do), quantiles=quantiles)
        results_extreme = triton.testing.do_bench(
            lambda: sparse_reward_propagation_triton(rewards_extreme, 0.99).backward(do), quantiles=quantiles)

    print(f"\n📌 Benchmark Results for batch_size={batch_size}, provider={provider}")

    print("\n📌 **Performance for 5% Sparsity**:")
    print(f"Execution Time (Median): {results_standard[0]} ms")
    print(f"Execution Time (20th Percentile): {results_standard[1]} ms")
    print(f"Execution Time (80th Percentile): {results_standard[2]} ms")

    print("\n📌 **Performance for 0.1% (Extreme Case) Sparsity**:")
    print(f"Execution Time (Median): {results_extreme[0]} ms")
    print(f"Execution Time (20th Percentile): {results_extreme[1]} ms")
    print(f"Execution Time (80th Percentile): {results_extreme[2]} ms")

if __name__ == '__main__':
    assert torch.cuda.is_available(), "❌ CUDA GPU is not available! Check your environment."
    print("🚀 GPU Check Passed. Device:", torch.cuda.get_device_name(0))

    benchmark.run(print_data=True)
