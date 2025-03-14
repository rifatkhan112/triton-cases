import torch
import triton
import triton.language as tl

@triton.jit
def sparse_propagate_kernel(
    rewards_ptr, dones_ptr, output_ptr,
    B, S, discount,
    rewards_stride_b, rewards_stride_s,
    dones_stride_b, dones_stride_s,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    for idx in range(pid, B * S, num_pids):
        b = idx // S
        s = idx % S
        
        # Load current reward and done status
        reward = tl.load(rewards_ptr + b*rewards_stride_b + s*rewards_stride_s)
        done = tl.load(dones_ptr + b*dones_stride_b + s*dones_stride_s)
        
        # Modified to avoid continue statement
        if not (reward == 0.0 and done == 0):
            # Find trajectory start
            start = s
            while start > 0:
                prev_done = tl.load(dones_ptr + b*dones_stride_b + (start-1)*dones_stride_s)
                if prev_done != 0:
                    break
                start -= 1
                
            # Propagate reward backward through trajectory
            cumulative = reward
            tl.atomic_add(output_ptr + b*rewards_stride_b + s*rewards_stride_s, cumulative)
            
            for t in range(s-1, start-1, -1):
                cumulative *= discount
                tl.atomic_add(output_ptr + b*rewards_stride_b + t*rewards_stride_s, cumulative)

def sparse_reward_propagation_triton(
    rewards: torch.Tensor,
    discount: float = 0.99,
    dones: torch.Tensor = None
) -> torch.Tensor:
    """Optimized Triton implementation with fixed control flow"""
    B, S = rewards.shape
    output = torch.zeros_like(rewards)
    
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.float32)
    
    def grid(meta): return (triton.cdiv(B * S, meta['BLOCK_SIZE']),)
    
    sparse_propagate_kernel[grid](
        rewards, dones, output,
        B, S, discount,
        rewards.stride(0), rewards.stride(1),
        dones.stride(0), dones.stride(1),
        BLOCK_SIZE=256
    )
    
    return output
