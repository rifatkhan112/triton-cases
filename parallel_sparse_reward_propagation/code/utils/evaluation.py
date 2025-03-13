import torch
import time
import numpy as np
from code.naive_implementation import sparse_reward_propagation_naive
from code.triton_implementation import sparse_reward_propagation_triton

def relative_error(A, B):
    return torch.norm(A - B) / torch.norm(A)

def evaluate_sparse_reward_propagation(batch_sizes=[1024, 2048, 4096], sequence_lengths=[50, 100, 200], discount=0.99):
    """
    Evaluates the performance and correctness of the Triton kernel against the naive PyTorch implementation.
    """
    device = "cuda"
    results = []

    for B in batch_sizes:
        for S in sequence_lengths:
            print(f"\n🔹 Evaluating B={B}, S={S}")

            # Initialize input tensors
            states = torch.randn((B, S), dtype=torch.float32, device=device, requires_grad=True)
            rewards = torch.zeros((B, S), dtype=torch.float32, device=device, requires_grad=True)

            # Introduce sparsity (5% non-zero rewards)
            mask = torch.rand_like(rewards) < 0.05
            rewards[mask] = torch.randn_like(rewards[mask])

            do = torch.randn_like(rewards)

            # ✅ Naive Implementation
            torch.cuda.synchronize()
            start_time = time.time()
            ref = sparse_reward_propagation_naive(states.cpu(), rewards.cpu()).cuda()
            torch.cuda.synchronize()
            naive_time = time.time() - start_time

            # ✅ Triton Implementation
            torch.cuda.synchronize()
            start_time = time.time()
            tri = sparse_reward_propagation_triton(states, rewards, discount=discount)
            torch.cuda.synchronize()
            triton_time = time.time() - start_time

            # ✅ Compute Gradient Differences
            ref.backward(do, retain_graph=True)
            tri.backward(do, retain_graph=True)

            ref_dstates, ref_drewards = states.grad.clone(), rewards.grad.clone()
            states.grad, rewards.grad = None, None

            tri_dstates, tri_drewards = states.grad.clone(), rewards.grad.clone()

            # ✅ Numerical Checks
            output_error = relative_error(ref, tri).item()
            gradient_error_states = relative_error(ref_dstates, tri_dstates).item()
            gradient_error_rewards = relative_error(ref_drewards, tri_drewards).item()

            # ✅ Performance Speedup
            speedup = naive_time / triton_time

            # ✅ Store Results
            results.append({
                "Batch Size": B,
                "Sequence Length": S,
                "Naive Time (s)": naive_time,
                "Triton Time (s)": triton_time,
                "Speedup": speedup,
                "Output Error": output_error,
                "Gradient Error (States)": gradient_error_states,
                "Gradient Error (Rewards)": gradient_error_rewards
            })

            print(f"✅ Speedup: {speedup:.2f}x (Naive: {naive_time:.4f}s, Triton: {triton_time:.4f}s)")
            print(f"🔍 Output Error: {output_error:.5f}")
            print(f"🔍 Gradient Error (States): {gradient_error_states:.5f}")
            print(f"🔍 Gradient Error (Rewards): {gradient_error_rewards:.5f}")

    return results

if __name__ == "__main__":
    results = evaluate_sparse_reward_propagation()
    # Save results as CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("\n📊 Evaluation complete! Results saved to 'evaluation_results.csv'.")