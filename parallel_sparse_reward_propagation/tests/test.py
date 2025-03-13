import torch
import torch.nn.functional as F
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def check_close(A, B):
    """ Check if two tensors are close within a numerical tolerance. """
    return torch.allclose(A, B, rtol=1e-3, atol=1e-5)

if __name__ == "__main__":
    # Define batch size and state-space size
    B, S = 4, 4096  # 4 batches, 4096 state transitions
    dtype = torch.float32

    # Initialize rewards tensor with gradients enabled
    rewards = torch.zeros((B, S), dtype=dtype, device="cuda", requires_grad=True)

    # Mask for sparse rewards (only 5% of states receive rewards)
    mask = torch.rand((B, S), device="cuda") < 0.05

    # Fix: Ensure in-place modification does not affect a leaf variable
    rewards = rewards.clone()
    rewards[mask] = torch.randn_like(rewards[mask]).detach()

    # Initialize state transitions and importance weights
    transitions = torch.randn((B, S, S), dtype=dtype, device="cuda", requires_grad=True)
    importance_weights = torch.ones((B, S), dtype=dtype, device="cuda", requires_grad=True)

    # Initialize random gradients for backpropagation test
    do = torch.randn_like(rewards)

    # Compute reference reward propagation (naive)
    ref_output = sparse_reward_propagation_naive(rewards, transitions, importance_weights)

    # Retain gradients and backpropagate
    rewards.retain_grad()
    ref_output.backward(do, retain_graph=True)

    # Store gradients
    ref_d_rewards, rewards.grad = rewards.grad.clone(), None
    ref_d_transitions, transitions.grad = transitions.grad.clone(), None
    ref_d_importance_weights, importance_weights.grad = importance_weights.grad.clone(), None

    # Compute Triton-based reward propagation
    tri_output = sparse_reward_propagation_triton(rewards, transitions, importance_weights)

    # Retain gradients and backpropagate
    rewards.retain_grad()
    tri_output.backward(do, retain_graph=True)

    # Store gradients
    tri_d_rewards, rewards.grad = rewards.grad.clone(), None
    tri_d_transitions, transitions.grad = transitions.grad.clone(), None
    tri_d_importance_weights, importance_weights.grad = importance_weights.grad.clone(), None

    # Assertions to ensure numerical equivalence
    assert check_close(ref_output, tri_output), "Output mismatch between naive and Triton implementation"
    assert check_close(ref_d_rewards, tri_d_rewards), "Gradient mismatch for rewards"
    assert check_close(ref_d_transitions, tri_d_transitions), "Gradient mismatch for transitions"
    assert check_close(ref_d_importance_weights, tri_d_importance_weights), "Gradient mismatch for importance weights"

    print("✅ Triton and Naive implementations match correctly!")
    print(f"⚡ Speedup: {naive_time / triton_time:.2f}x (Naive: {naive_time:.4f}s, Triton: {triton_time:.4f}s)")
