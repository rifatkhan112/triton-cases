import torch
import torch.nn.functional as F
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def check_close(A, B, atol=1e-5):
    """Check if two tensors are close within a tolerance."""
    is_close = torch.allclose(A, B, rtol=0, atol=atol)
    if not is_close:
        print(f"Max difference: {torch.max(torch.abs(A - B))}")
    return is_close

if __name__ == "__main__":
    # Define batch size, sequence length
    B, S = 4, 4096  
    dtype = torch.float32  # Ensure consistency

    # Generate random input tensors
    rewards = torch.randn((B, S), dtype=dtype, device="cuda", requires_grad=True)

    # Ensure `transitions` is a valid transition matrix (B, S, S) if needed
    transitions = torch.randint(0, 2, (B, S, S), dtype=torch.float32, device="cuda")
    transitions /= transitions.sum(dim=-1, keepdim=True)  # Normalize for valid transitions

    importance_weights = torch.rand((B, S), dtype=dtype, device="cuda", requires_grad=True)

    # Compute outputs using naive and triton implementations
    ref_output = sparse_reward_propagation_naive(rewards, transitions, importance_weights)
    discount_factor = 0.99  # Example value, adjust as needed
    tri_output = sparse_reward_propagation_triton(rewards, transitions, importance_weights, discount_factor)

    # Print shape info to debug
    print(f"ref_output shape: {ref_output.shape}")
    print(f"tri_output shape: {tri_output.shape}")

    # Check if outputs match
    assert check_close(ref_output, tri_output), "Triton and naive implementations do NOT match!"

    # Define gradient tensor with correct shape
    do = torch.randn_like(ref_output, dtype=dtype)

    # Ensure gradient tensor matches output shape
    print(f"do shape before fix: {do.shape}")
    if do.shape != ref_output.shape:
        do = do.expand_as(ref_output)  # Ensure `do` matches ref_output
    print(f"do shape after fix: {do.shape}")

    # Compute gradients
    ref_output.backward(do, retain_graph=True)
    tri_output.backward(do, retain_graph=True)

    # Extract gradients safely
    ref_d_rewards, ref_d_weights = rewards.grad.clone(), importance_weights.grad.clone()

    # Reset gradients safely
    rewards.grad.zero_()
    importance_weights.grad.zero_()

    tri_d_rewards, tri_d_weights = rewards.grad.clone(), importance_weights.grad.clone()

    # Check if gradients match
    assert check_close(ref_d_rewards, tri_d_rewards), "Gradient mismatch in rewards!"
    assert check_close(ref_d_weights, tri_d_weights), "Gradient mismatch in importance weights!"

    print("✅ Triton and Naive outputs & gradients match! Test Passed.")
    assert check_close(ref_d_rewards, tri_d_rewards), "Gradient mismatch in rewards!"
    assert check_close(ref_d_weights, tri_d_weights), "Gradient mismatch in importance weights!"

    print("✅ Triton and Naive outputs & gradients match! Test Passed.")
