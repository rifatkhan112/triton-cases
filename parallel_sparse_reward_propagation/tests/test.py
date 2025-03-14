import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def check_close(A, B, atol=1e-5):
    """Check if two tensors are close with a given absolute tolerance."""
    is_close = torch.allclose(A, B, rtol=0, atol=atol)
    if not is_close:
        print("Max difference:", (A - B).abs().max().item())
    return is_close

if __name__ == "__main__":
    # ✅ Test parameters
    B, S, K = 4, 4096, 5  # Batch size, sequence length, sparse reward count
    discount_factor = 0.99

    torch.manual_seed(42)

    # ✅ Generate sparse rewards
    rewards = torch.zeros((B, S), dtype=torch.float32, device="cuda", requires_grad=True)
    sparse_indices = torch.randint(0, S, (B, K), dtype=torch.int32, device="cuda")

    # Assign random sparse rewards
    for b in range(B):
        rewards[b, sparse_indices[b]] = torch.randn(K, device="cuda")

    # ✅ Create independent copies for gradient tests
    rewards_naive = rewards.clone().detach().requires_grad_()
    rewards_triton = rewards.clone().detach().requires_grad_()

    # ✅ Run naive implementation
    out_naive = sparse_reward_propagation_naive(rewards_naive, sparse_indices, discount=discount_factor)

    # ✅ Run Triton implementation
    out_triton = sparse_reward_propagation_triton(rewards_triton, sparse_indices, discount=discount_factor)

    # ✅ Compare forward results
    print("\n🚀 **Sparse Reward Propagation Test** 🚀")
    print("ref_output shape:", out_naive.shape)
    print("tri_output shape:", out_triton.shape)
    print("Check forward match:", check_close(out_naive, out_triton))

    # ✅ Compute gradients for backpropagation verification
    grad_out = torch.ones_like(out_naive)

    out_naive.backward(grad_out, retain_graph=True)
    out_triton.backward(grad_out, retain_graph=True)

    grad_naive = rewards_naive.grad.clone()
    grad_triton = rewards_triton.grad.clone()

    # ✅ Compare gradients
    print("grad_naive shape:", grad_naive.shape)
    print("grad_triton shape:", grad_triton.shape)
    print("Check gradient match:", check_close(grad_naive, grad_triton))

    print("\n✅ Done. All tests completed.")
