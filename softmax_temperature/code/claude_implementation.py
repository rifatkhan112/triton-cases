import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
from torch.utils.benchmark import Timer

# Naive PyTorch implementation for comparison
def naive_softmax(x, temperature=100.0):
    x_temp = x / temperature
    # Compute max for numerical stability
    x_max = torch.max(x_temp, dim=1, keepdim=True)[0]
    numerator = torch.exp(x_temp - x_max)
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    return numerator / denominator

# Triton implementation
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 2048}),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 1024}),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 512}),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}),
    ],
    key=['M', 'N'],
)
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, temp,
    stride_om, stride_on,  # output strides
    stride_im, stride_in,  # input strides
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Row index
    row_id = tl.program_id(0)
    row_offset = row_id * BLOCK_SIZE_M
    
    # Column indices
    cols = tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid columns
    mask = cols < N
    
    # Per-row loop
    for m in range(row_offset, min(row_offset + BLOCK_SIZE_M, M)):
        # Compute input and output pointers for this row
        row_input_ptr = input_ptr + m * stride_im
        row_output_ptr = output_ptr + m * stride_om
        
        # Load input row with masking
        row = tl.load(row_input_ptr + cols * stride_in, mask=mask, other=-float('inf'))
        
        # Apply temperature
        row = row / temp
        
        # Compute max for numerical stability
        row_max = tl.max(row, axis=0)
        
        # Compute exponentials
        row = row - row_max
        numerator = tl.exp(row)
        
        # Compute sum of exponentials
        denominator = tl.sum(numerator, axis=0)
        
        # Normalize
        softmax_output = numerator / denominator
        
        # Store output row
        tl.store(row_output_ptr + cols * stride_on, softmax_output, mask=mask)

# Wrapper function
def softmax_with_temp(x, temperature=100.0):
    """
    Apply softmax with temperature to the input tensor.
    
    Args:
        x: Input tensor of shape (M, N)
        temperature: Temperature parameter (default: 100.0)
    
    Returns:
        Output tensor of shape (M, N)
    """
    M, N = x.shape
    y = torch.empty_like(x)
    
    # Launch kernel
    grid = (triton.cdiv(M, 1),)
    softmax_kernel[grid](
        y, x, temperature,
        y.stride(0), y.stride(1),
        x.stride(0), x.stride(1),
        M, N,
        1, N,  # The block size will be autotuned
    )
    return y

# Benchmark function to compare naive vs Triton implementation
def benchmark(M=1823, N=781, warmup=25, rep=100):
    x = torch.randn((M, N), device='cuda', dtype=torch.float32)
    
    # Record execution times
    naive_timing = Timer(
        stmt='naive_softmax(x, temperature)',
        globals={'naive_softmax': naive_softmax, 'x': x, 'temperature': 100.0}
    )
    triton_timing = Timer(
        stmt='softmax_with_temp(x, temperature)',
        globals={'softmax_with_temp': softmax_with_temp, 'x': x, 'temperature': 100.0}
    )
    
    naive_time = naive_timing.timeit(warmup, rep)
    triton_time = triton_timing.timeit(warmup, rep)
    
    # Verify results
    naive_out = naive_softmax(x, 100.0)
    triton_out = softmax_with_temp(x, 100.0)
    max_diff = torch.max(torch.abs(naive_out - triton_out)).item()
    
    # Calculate speedup
    speedup = naive_time / triton_time
    
    return {
        'shape': (M, N),
        'naive_time_ms': naive_time * 1000,
        'triton_time_ms': triton_time * 1000,
        'speedup': speedup,
        'max_diff': max_diff
    }

# Run benchmark
if __name__ == "__main__":
    results = benchmark(M=1823, N=781)
    print(f"Shape: {results['shape']}")
    print(f"Naive implementation: {results['naive_time_ms']:.3f} ms")
    print(f"Triton implementation: {results['triton_time_ms']:.3f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"Max difference: {results['max_diff']:.2e}")
