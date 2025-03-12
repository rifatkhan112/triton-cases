import torch
import time
from code.naive_implementation import sparse_reward_propagation_naive
from code.triton_implementation import sparse_reward_propagation_triton
from code.utils.data_loader import load_sparse_transitions

def benchmark():
    states, rewards = load_sparse_transitions(batch_size=10000)
    
    # Benchmark Naive CPU Implementation
    start = time.time()
    sparse_reward_propagation_naive(states, rewards)
    cpu_time = time.time() - start
    print(f"Naive CPU Execution Time: {cpu_time:.6f} sec")
    
    # Benchmark Triton GPU Implementation
    states, rewards = states.cuda(), rewards.cuda()
    start = time.time()
    sparse_reward_propagation_triton(states, rewards)
    gpu_time = time.time() - start
    print(f"Triton GPU Execution Time: {gpu_time:.6f} sec")
    
    print(f"✅ Speedup Achieved: {cpu_time / gpu_time:.2f}x (Expected: 6-7x)")

if __name__ == "__main__":
    benchmark()