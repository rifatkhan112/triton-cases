import torch
from code.naive_implementation import sparse_reward_propagation_naive
from code.triton_implementation import sparse_reward_propagation_triton
from code.utils.data_loader import load_sparse_transitions
from code.utils.evaluation import compute_mse

def test_sparse_reward_propagation():
    states, rewards = load_sparse_transitions(batch_size=10)
    cpu_result = sparse_reward_propagation_naive(states, rewards)
    gpu_result = sparse_reward_propagation_triton(states.cuda(), rewards.cuda()).cpu()
    
    mse = compute_mse(cpu_result, gpu_result)
    assert mse < 1e-5, f"MSE too high: {mse}"
    print("✅ Test Passed: Triton matches Naive CPU Implementation")

if __name__ == "__main__":
    test_sparse_reward_propagation()