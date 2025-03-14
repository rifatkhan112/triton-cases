import torch
import triton
import triton.language as tl

@triton.jit
def optimized_sparse_propagate(
    rewards_ptr, dones_ptr, output_ptr,
    B, S, gamma,
    reward_stride_b, reward_stride_s,
    done_stride_b, done_stride_s,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # Process multiple elements per thread
    for idx in range(pid, B * S, num_pids):
        b = idx // S
        s = idx % S
        
        # Load current reward and done status
        reward = tl.load(rewards_ptr + b*reward_stride_b + s*reward_stride_s)
        done = tl.load(dones_ptr + b*done_stride_b + s*done_stride_s)
        
        if reward == 0.0 and done == 0:
            continue  # Skip non-terminal, zero-reward steps
            
        # Find trajectory start
        start = s
        while start > 0:
            prev_done = tl.load(dones_ptr + b*done_stride_b + (start-1)*done_stride_s)
            if prev_done != 0:
                break
            start -= 1
            
        # Propagate reward backward through trajectory
        cumulative = reward
        tl.atomic_add(output_ptr + b*reward_stride_b + s*reward_stride_s, cumulative)
        
        for t in range(s-1, start-1, -1):
            cumulative *= gamma
            tl.atomic_add(output_ptr + b*reward_stride_b + t*reward_stride_s, cumulative)

def hybrid_propagate(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Batched sparse reward propagation with Triton optimization
    Args:
        rewards: [B, S] tensor of sparse rewards
        dones: [B, S] tensor of episode termination flags
        gamma: discount factor
    Returns:
        [B, S] tensor of propagated returns
    """
    B, S = rewards.shape
    output = torch.zeros_like(rewards)
    
    # Configure kernel launch parameters
    def grid(meta): return (triton.cdiv(B * S, meta['BLOCK_SIZE']),)
    
    optimized_sparse_propagate[grid](
        rewards, dones, output,
        B, S, gamma,
        rewards.stride(0), rewards.stride(1),
        dones.stride(0), dones.stride(1),
        BLOCK_SIZE=256
    )
    
    return output
