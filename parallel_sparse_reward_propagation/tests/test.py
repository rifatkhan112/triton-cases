import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def check_close(A, B, atol=1e-5):
    is_close = torch.allclose(A, B, rtol=0, atol=atol)
    if not is_close:
        print("Max diff:", (A - B).abs().max().item())
    return is_close

if __name__ == "__main__":
    B, S, K = 4, 4096, 205  # ~5% sparsity
    discount_factor = 0.99

    rewards_naive = torch.randn((B, S), device="cuda", requires_grad=True)
    rewards_triton = rewards_naive.clone().detach().requires_grad_()

    sparse_indices = torch.randint(0, S-1, (B, K), device="cuda")

    # Forward pass
    out_naive = sparse_reward_propagation_naive(rewards_naive, sparse_indices, discount=discount_factor)
    out_triton = sparse_reward_propagation_triton(rewards_triton, sparse_indices, discount=discount_factor)

    print("Naive output shape:", out_naive.shape)
    print("Triton output shape:", out_triton.shape)

    # Check forward correctness
    forward_match = check_close(out_naive, out_triton)
    print("Forward pass match:", forward_match)

    # Backward pass
    grad_output = torch.ones_like(out_naive)
    out_naive.backward(grad_output, retain_graph=True)
    out_triton.backward(grad_output, retain_graph=True)

    grad_naive = rewards_naive.grad.clone()
    grad_triton = rewards_triton.grad.clone()

    # Check backward correctness
    grad_match = check_close(grad_naive, grad_triton)
    print("Backward pass match:", grad_match)

    print("Done.")
