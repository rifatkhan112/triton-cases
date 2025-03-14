import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

def test_forward():
    """Validate forward pass consistency"""
    B, S = 4, 4096
    rewards = torch.randn(B, S, device="cuda")
    dones = torch.bernoulli(torch.full((B, S), 0.1, device="cuda"))
    
    ref = sparse_reward_propagation_naive(rewards, discount=0.99, dones=dones)
    tri = sparse_reward_propagation_triton(rewards, discount=0.99, dones=dones)
    
    assert torch.allclose(ref, tri, atol=1e-5), f"Max diff: {(ref - tri).abs().max().item()}"

def test_gradients():
    """Verify gradient calculations"""
    B, S = 2, 128
    rewards = torch.randn(B, S, device="cuda", requires_grad=True)
    dones = torch.bernoulli(torch.full((B, S), 0.1, device="cuda"))
    
    for impl in [sparse_reward_propagation_naive, sparse_reward_propagation_triton]:
        out = impl(rewards.clone(), 0.99, dones)
        grad = torch.autograd.grad(out.sum(), rewards, retain_graph=True)[0]
        assert not torch.isnan(grad).any(), "NaN values in gradients"

if __name__ == "__main__":
    test_forward()
    test_gradients()
    print("All tests passed successfully!")
