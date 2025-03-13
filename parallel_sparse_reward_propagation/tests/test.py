import torch
import torch.nn.functional as F
import time
from code.naive_implementation import sparse_reward_propagation_naive
from code.triton_implementation import sparse_reward_propagation_triton

def check_close(A, B, atol=1e-3):
    return torch.allclose(A, B, rtol=0, atol=atol)

if __name__ == "__main__":
    B, S = 4096, 100  # Batch size and sequence length
    dtype = torch.float32

    # Initialize state transitions and sparse rewards
    states = torch.randn((B, S), dtype=dtype, device="cuda", requires_grad=True)
    rewards = torch.zeros((B, S), dtype=dtype, device="cuda", requires_grad=True)
    
    # Introduce sparsity (only 5% non-zero rewards)
    mask = torch.rand_like(rewards) < 0.05
    rewards[mask] = torch.randn_like(rewards[mask])

    do = torch.randn_like(rewards)  # Dummy gradient

    # Compute reference output (Naive CPU version)
    start_time = time.time()
    ref = sparse_reward_propagation_naive(states.cpu(), rewards.cpu()).cuda()
    naive_time = time.time() - start_time

    # Compute Triton kernel output
    start_time = time.time()
    tri = sparse_reward_propagation_triton(states, rewards)
    triton_time = time.time() - start_time

    # Retain gradients
    states.retain_grad(), rewards.retain_grad()
    
    # Compute gradients for naive implementation
    ref.backward(do, retain_graph=True)
    ref_dstates, states.grad = states.grad.clone(), None
    ref_drewards, rewards.grad = rewards.grad.clone(), None

    # Compute gradients for Triton implementation
    tri.backward(do, retain_graph=True)
    tri_dstates, states.grad = states.grad.clone(), None
    tri_drewards, rewards.grad = rewards.grad.clone(), None

    # Validate correctness
    assert check_close(ref, tri), "Output mismatch between Naive and Triton implementations!"
    assert check_close(ref_dstates, tri_dstates), "Gradient mismatch in states!"
    assert check_close(ref_drewards, tri_drewards), "Gradient mismatch in rewards!"

    print(f"✅ Triton and PyTorch implementations match!")
    print(f"⚡ Speedup: {naive_time / triton_time:.2f}x (Naive: {naive_time:.4f}s, Triton: {triton_time:.4f}s)")