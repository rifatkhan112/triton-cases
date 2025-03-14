import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def check_close(A, B, atol=1e-5):
    is_close = torch.allclose(A, B, atol=atol)
    if not is_close:
        print("Max diff:", (A - B).abs().max())
    return is_close

if __name__ == "__main__":
    B, S = 4, 4096
    discount = 0.99

    rewards_naive = torch.randn((B, S), device='cuda', requires_grad=True)
    rewards_triton = rewards_naive.clone().detach().requires_grad_()

    # Forward
    out_naive = sparse_reward_propagation_naive(rewards_naive, discount)
    out_triton = sparse_reward_propagation_triton(rewards_triton, discount)

    print("Forward match:", check_close(out_naive, out_triton))

    # Backward
    grad_out = torch.ones_like(out_naive)
    out_naive.backward(grad_out, retain_graph=True)
    out_triton.backward(grad_out, retain_graph=True)

    grad_naive = rewards_naive.grad
    grad_triton = rewards_triton.grad

    print("Gradient match:", check_close(grad_naive, grad_triton))
