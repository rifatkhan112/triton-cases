import torch
import triton
import triton.language as tl

# Optimized Triton kernel for softmax with temperature parameter
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(
    x_ptr,               # pointer to input tensor
    output_ptr,          # pointer to output tensor
    temp,                # temperature parameter (100.0)
    n_cols,              # number of columns in the input
    stride_row,          # stride between rows in the input
    stride_col,          # stride between columns in the input
    output_stride_row,   # stride between rows in the output
    output_stride_col,   # stride between columns in the output
    BLOCK_SIZE: tl.constexpr,  # block size for tiling
):
    # Get program ID (row index)
    row_idx = tl.program_id(0)
    
    # Compute the pointers to this row
    row_start_ptr = x_ptr + row_idx * stride_row
    output_row_start_ptr = output_ptr + row_idx * output_stride_row
    
    # Create column offsets for processing blocks of columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize variables to compute the softmax
    row_max = tl.zeros([1], dtype=tl.float32) - float("inf")
    
    # First pass: find the maximum value in the row (for numerical stability)
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create mask for valid columns (to handle edge cases)
        mask = col_offsets < n_cols - col_start
        
        # Load data for this block - ensure memory coalescing with stride_col
        col_ptrs = row_start_ptr + (col_start + col_offsets) * stride_col
        block_data = tl.load(col_ptrs, mask=mask, other=float("-inf"))
        
        # Update the running maximum
        block_max = tl.max(block_data, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # Initialize sum for the softmax denominator
    exp_sum = tl.zeros([1], dtype=tl.float32)
    
    # Second pass: compute the sum of exp((x - max) / temp)
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create mask for valid columns
        mask = col_offsets < n_cols - col_start
        
        # Load data for this block
        col_ptrs = row_start_ptr + (col_start + col_offsets) * stride_col
        block_data = tl.load(col_ptrs, mask=mask, other=float("-inf"))
        
        # Apply exp((x - max) / temp) for numerical stability
        block_exp = tl.exp((block_data - row_max) / temp)
        
        # Update the running sum
        block_sum = tl.sum(block_exp, axis=0)
        exp_sum += block_sum
    
    # Third pass: compute the softmax output
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create mask for valid columns
        mask = col_offsets < n_cols - col_start
        
        # Load data for this block
        col_ptrs = row_start_ptr + (col_start + col_offsets) * stride_col
        block_data = tl.load(col_ptrs, mask=mask, other=float("-inf"))
        
        # Compute exp((x - max) / temp) / sum(exp((x - max) / temp))
        output_block = tl.exp((block_data - row_max) / temp) / exp_sum
        
        # Store the results with optimal memory layout
        output_ptrs = output_row_start_ptr + (col_start + col_offsets) * output_stride_col
        tl.store(output_ptrs, output_block, mask=mask)

# Wrapper function to queue the kernel with appropriate meta-parameters
def softmax_with_temperature(x, temperature=100.0):
    """
    Apply softmax with temperature to a tensor, optimized for A100 GPU.
    
    Args:
        x: Input tensor of shape (batch_size, n_features)
        temperature: Temperature parameter (default: 100.0)
        
    Returns:
        Tensor of the same shape as x with softmax applied along the last dimension
    """
    # Get input shape and create output tensor
    batch_size, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Compute strides for memory access
    stride_row, stride_col = x.stride()
    output_stride_row, output_stride_col = output.stride()
    
    # Launch the kernel
    grid = (batch_size,)
    softmax_kernel[grid](
        x, output, temperature, n_cols,
        stride_row, stride_col,
        output_stride_row, output_stride_col,
    )
    
    return output

# Benchmark function to compare with PyTorch implementation
def benchmark_softmax(batch_size=1823, n_features=781, temperature=100.0, num_runs=100):
    """
    Benchmark the Triton softmax implementation against PyTorch's native implementation.
    
    Args:
        batch_size: Number of rows in the input tensor (default: 1823)
        n_features: Number of columns in the input tensor (default: 781)
        temperature: Temperature parameter for softmax (default: 100.0)
        num_runs: Number of runs for benchmarking
        
    Returns:
        Average time for each implementation and speedup factor
    """
    # Create random input tensor on GPU
    x = torch.randn((batch_size, n_features), device='cuda')
    
    # PyTorch native implementation (naive approach)
    def pytorch_softmax():
        return torch.nn.functional.softmax(x / temperature, dim=1)
    
    # Triton implementation
    def triton_softmax():
        return softmax_with_temperature(x, temperature)
    
    # Warm-up runs to avoid initial overhead
    for _ in range(10):
        pytorch_result = pytorch_softmax()
        triton_result = triton_softmax()
    
    # Verify correctness by comparing results
    pytorch_result = pytorch_softmax()
    triton_result = triton_softmax()
    assert torch.allclose(pytorch_result, triton_result, rtol=1e-3, atol=1e-3), \
        "Results do not match!"
    
    # Benchmark PyTorch implementation
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        pytorch_result = pytorch_softmax()
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / num_runs
    
    # Benchmark Triton implementation
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        triton_result = triton_softmax()
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_runs
    
    # Calculate speedup
    speedup = pytorch_time / triton_time
    
    print(f"PyTorch time: {pytorch_time:.4f} ms")
    print(f"Triton time: {triton_time:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return pytorch_time, triton_time, speedup

# Example usage
if __name__ == "__main__":
    # Benchmark for the specified tensor size (1823, 781)
    pytorch_time, triton_time, speedup = benchmark_softmax(
        batch_size=1823, 
        n_features=781, 
        temperature=100.0
    )
    
    print(f"Expected speedup: 6.00x")
    print(f"Actual speedup: {speedup:.2f}x")
