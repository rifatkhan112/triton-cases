import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def check_close(A, B, atol=1e-5):
    """Check if two tensors are close within a tolerance."""
    is_close = torch.allclose(A, B, rtol=0, atol=atol)
    if not is_close:
        diff = torch.max(torch.abs(A - B)).item()
        print(f"[check_close] Max difference: {diff}")
    return is_close

if __name__ == "__main__":
    # Example config
    B, S = 4, 4096
    discount_factor = 0.99

    # Create random rewards
    rewards = torch.randn((B, S), device="cuda", dtype=torch.float32, requires_grad=True)
    rewards_copy = rewards.clone().detach().requires_grad_()

    # Naive approach
    ref_output = sparse_reward_propagation_naive(rewards, discount=discount_factor)

    # Triton approach
    tri_output = sparse_reward_propagation_triton(rewards_copy, discount=discount_factor)

    # Check outputs match
    assert check_close(ref_output, tri_output), "❌ Outputs do not match!"

    # Create grad outputs for backward
    grad_out = torch.randn_like(ref_output)

    # Backward pass on both
    ref_output.backward(grad_out, retain_graph=True)
    tri_output.backward(grad_out, retain_graph=True)

    # Compare gradients
    grad_naive = rewards.grad.clone()
    grad_triton = rewards_copy.grad.clone()

    # Check if grads match
    assert check_close(grad_naive, grad_triton), "❌ Gradients do not match!"

    print("✅ All tests passed. Triton and naive results match!")
