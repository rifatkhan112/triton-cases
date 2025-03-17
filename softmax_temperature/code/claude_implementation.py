import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, 
    stride_om, stride_on,
    stride_im, stride_in,
    n_cols, temperature,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute softmax with temperature: softmax(x/temperature)
    
    output_ptr: pointer to output tensor
    input_ptr: pointer to input tensor
    stride_om, stride_on: strides for output tensor
    stride_im, stride_in: strides for input tensor
    n_cols: number of columns in input
    temperature: softmax temperature parameter
    BLOCK_SIZE: number of elements to process per block
    """
    # Get program ID
    row_idx = tl.program_id(0)
    
    # Compute pointers
    row_start_ptr = input_ptr + row_idx * stride_im
    out_row_start_ptr = output_ptr + row_idx * stride_om
    
    # Initialize maximum value for numerical stability
    row_max = tl.zeros([1], dtype=tl.float32) - float("inf")
    
    # First pass: find max
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create block mask for handling boundaries
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data for this block
        x = tl.load(row_start_ptr + col_offsets * stride_in, mask=mask, other=-float("inf"))
        
        # Update max
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Second pass: compute exponentials with temperature and sum
    row_sum = tl.zeros([1], dtype=tl.float32)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create block mask for handling boundaries
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data for this block
        x = tl.load(row_start_ptr + col_offsets * stride_in, mask=mask, other=-float("inf"))
        
        # Apply temperature and compute exp(x - max) for numerical stability
        x = tl.exp((x - row_max) / temperature)
        
        # Update sum
        row_sum += tl.sum(x, axis=0)
        
        # Store partial softmax results
        tl.store(out_row_start_ptr + col_offsets * stride_on, x, mask=mask)
    
    # Third pass: normalize by sum
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create block mask for handling boundaries
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data for this block
        x = tl.load(out_row_start_ptr + col_offsets * stride_on, mask=mask)
        
        # Normalize
        x = x / row_sum
        
        # Store normalized results
        tl.store(out_row_start_ptr + col_offsets * stride_on, x, mask=mask)


def softmax_with_temperature(x: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
    """
    Apply softmax with temperature parameter using Triton kernel
    
    Args:
        x: Input tensor of shape (M, N)
        temperature: Temperature parameter for softmax (default: 100.0)
        
    Returns:
        Softmax output of same shape as input
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    
    # Output tensor
    o = torch.empty_like(x)
    
    # Get shapes and strides
    M, N = x.shape
    stride_im, stride_in = x.stride()
    stride_om, stride_on = o.stride()
    
    # Determine optimal block size based on input dimensions
    # These values are tuned for A100 GPUs
    if N <= 256:
        BLOCK_SIZE = 128
    elif N <= 512:
        BLOCK_SIZE = 256
    elif N <= 1024:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
        
    # For our specific target size (1823, 781)
    if M == 1823 and N == 781:
        BLOCK_SIZE = 256  # Tuned specifically for this case

    # Launch kernel
    grid = (M,)
    softmax_kernel[grid](
        o, x,
        stride_om, stride_on,
        stride_im, stride_in,
        N, temperature,
        BLOCK_SIZE
    )
    
    return o


# Naive implementation for comparison
def naive_softmax_with_temperature(x: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
    """Standard PyTorch implementation of softmax with temperature"""
    return torch.softmax(x / temperature, dim=1)


# Benchmark function
def benchmark_softmax(M: int = 1823, N: int = 781, temperature: float = 100.0, num_runs: int = 100):
    """
    Benchmark Triton softmax against naive implementation
    
    Args:
        M: Number of rows
        N: Number of columns
        temperature: Temperature parameter
        num_runs: Number of benchmark runs
    """
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    # Verify correctness
    triton_output = softmax_with_temperature(x, temperature)
    naive_output = naive_softmax_with_temperature(x, temperature)
    print(f"Max error: {(triton_output - naive_output).abs().max().item()}")
    
    # Warmup
    for _ in range(10):
        naive_softmax_with_temperature(x, temperature)
        softmax_with_temperature(x, temperature)
    
    # Benchmark naive
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        naive_softmax_with_temperature(x, temperature)
    end.record()
    torch.cuda.synchronize()
    naive_time = start.elapsed_time(end) / num_runs
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        softmax_with_temperature(x, temperature)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_runs
    
    # Print results
    speedup = naive_time / triton_time
    print(f"Input shape: ({M}, {N})")
    print(f"Naive implementation: {naive_time:.3f} ms")
    print(f"Triton implementation: {triton_time:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup


if __name__ == "__main__":
    # Run benchmark for the specified shape
    speedup = benchmark_softmax(M=1823, N=781, temperature=100.0)
    print(f"Target speedup was 6x, achieved {speedup:.2f}x")
