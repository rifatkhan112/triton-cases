import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def check_close(A, B, atol=1e-5):
    is_close = torch.allclose(A, B, rtol=0, atol=atol)
    if not is_close:
        print("Max diff:", (A - B).abs().max().item())
    return is_close

if __name__ == "__main__":
    B, S = 4, 4096
    discount_factor = 0.99

    rewards = torch.randn((B, S), device="cuda", dtype=torch.float32, requires_grad=True)
    rewards_copy = rewards.clone().detach().requires_grad_()

    # Naive implementation
    ref_output = sparse_reward_propagation_naive(rewards, discount=discount_factor)

    # Triton implementation
    tri_output = sparse_reward_propagation_triton(rewards_copy := rewards.clone().detach().requires_grad_(), discount=discount_factor)

    print("Naive output shape:", ref_output.shape)
    print("Triton output shape:", tri_output.shape)

    # Forward check
    print("Forward pass match:", check_close(ref_output, tri_output))

    # Backward check
    grad_out = torch.ones_like(ref_output)
    ref_output.backward(grad_out, retain_graph=True)
    tri_output.backward(grad_out, retain_graph=True)

    grad_naive = rewards.grad.clone()
    grad_triton = rewards_copy.grad.clone()

    print("Gradient check:", check_close(grad_naive, grad_triton))

    print("Test complete.")
