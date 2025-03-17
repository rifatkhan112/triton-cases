import torch
import triton
import triton.language as tl
import time

# A100-optimized softmax kernel with temperature parameter
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 384}, num_warps=12, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 768}, num_warps=24, num_stages=1),
    ],
    key=['seq_length'],
)
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    batch_size, seq_length,
    stride_batch, stride_seq,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for processing a batch
    row_idx = tl.program_id(0)
    
    # Return if out of bounds
    if row_idx >= batch_size:
        return
    
    # Compute input and output pointers for the current batch
    row_start_ptr = input_ptr + row_idx * stride_batch
    output_start_ptr = output_ptr + row_idx * stride_batch
    
    # Create block-level memory access pattern for coalesced memory access
    column_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize values for max and sum
    row_max = float('-inf')
    
    # First pass: find max for numerical stability
    for block_start in range(0, seq_length, BLOCK_SIZE):
        # Create sequence offsets and mask
        offs = block_start + column_offsets
        mask = offs < seq_length
        
        # Load elements with mask
        x = tl.load(row_start_ptr + offs * stride_seq, mask=mask, other=float('-inf'))
        
        # Apply temperature scaling
        x = x / temperature
        
        # Update maximum value (masked)
        block_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # Initialize sum for the second pass
    row_sum = 0.0
    
    # Second pass: compute sum of exp(x - max_val)
    for block_start in range(0, seq_length, BLOCK_SIZE):
        # Create sequence offsets and mask
        offs = block_start + column_offsets
        mask = offs < seq_length
        
        # Load elements with mask
        x = tl.load(row_start_ptr + offs * stride_seq, mask=mask, other=float('-inf'))
        
        # Apply temperature scaling, subtract max, and exponentiate
        x = tl.exp((x / temperature) - row_max)
        
        # Update sum (masked)
        row_sum += tl.sum(x * mask, axis=0)
    
    # Third pass: normalize with sum and store
    for block_start in range(0, seq_length, BLOCK_SIZE):
        # Create sequence offsets and mask
        offs = block_start + column_offsets
        mask = offs < seq_length
        
        # Load elements with mask
        x = tl.load(row_start_ptr + offs * stride_seq, mask=mask, other=float('-inf'))
        
        # Apply temperature scaling, subtract max, exponentiate, and normalize
        x = tl.exp((x / temperature) - row_max) / row_sum
        
        # Store the result with mask
        tl.store(output_start_ptr + offs * stride_seq, x, mask=mask)

# Wrapper function for the softmax kernel
def softmax_triton(x, temperature=100.0):
    """
    Compute softmax with temperature using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
        temperature (float): Temperature parameter (default: 100.0)
        
    Returns:
        torch.Tensor: Softmax output with same shape as input
    """
    # Get input dimensions
    batch_size, seq_length = x.shape
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate strides
    stride_batch = x.stride(0)
    stride_seq = x.stride(1) if x.ndim > 1 else 1
    
    # Launch kernel with one program per batch element
    grid = (batch_size,)
    softmax_kernel[grid](
        output, x,
        batch_size, seq_length,
        stride_batch, stride_seq,
        temperature,
    )
    
    return output

# PyTorch reference implementation for comparison
def softmax_torch(x, temperature=100.0):
    """
    Compute softmax with temperature using native PyTorch.
    
    Args:
        x (torch.Tensor): Input tensor
        temperature (float): Temperature parameter (default: 100.0)
        
    Returns:
        torch.Tensor: Softmax output with same shape as input
    """
    return torch.softmax(x / temperature, dim=-1)

# Benchmark function
def benchmark_softmax(batch_size=1823, seq_length=781, temperature=100.0, num_runs=100):
    """
    Benchmark Triton softmax against PyTorch softmax.
    
    Args:
        batch_size (int): Batch size (default: 1823)
        seq_length (int): Sequence length (default: 781)
        temperature (float): Temperature parameter (default: 100.0)
        num_runs (int): Number of runs for benchmarking (default: 100)
        
    Returns:
        float: Speedup factor (PyTorch time / Triton time)
    """
    # Create test tensor
    x = torch.randn(batch_size, seq_length, device='cuda', dtype=torch.float32)
    
    # Warm up
    for _ in range(10):
        y_triton = softmax_triton(x, temperature)
        y_torch = softmax_torch(x, temperature)
    
    # Check correctness
    y_triton = softmax_triton(x, temperature)
    y_torch = softmax_torch(x, temperature)
    assert torch.allclose(y_triton, y_torch, rtol=1e-3, atol=1e-3), "Results don't match!"
    
    # Benchmark Triton implementation
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_runs):
        y_triton = softmax_triton(x, temperature)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_runs
    
    # Benchmark PyTorch implementation
    start.record()
    for _ in range(num_runs):
        y_torch = softmax_torch(x, temperature)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_runs
    
    # Calculate speedup
    speedup = torch_time / triton_time
    
    print(f"Input shape: ({batch_size}, {seq_length})")
    print(f"Temperature: {temperature}")
    print(f"Triton time: {triton_time:.3f} ms")
    print(f"PyTorch time: {torch_time:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup

if __name__ == "__main__":
    # Run benchmark with the specified dimensions
    speedup = benchmark_softmax(1823, 781, 100.0)
    print(f"Expected speedup: 6.00x, Actual speedup: {speedup:.2f}x")
