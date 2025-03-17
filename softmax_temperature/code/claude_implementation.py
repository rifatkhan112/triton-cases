import torch
import triton
import triton.language as tl

# Triton kernel for softmax with temperature, optimized for A100 GPUs
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, 
    input_row_stride, output_row_stride, 
    n_cols, temperature,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    # Get the program ID for the row
    row_idx = tl.program_id(0)
    
    # Compute the pointers to the input and output row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Create a mask for valid columns (in case n_cols is not a multiple of BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load the row with the mask - vectorized loads for better memory coalescing on A100
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Apply temperature scaling (temperature=100.0)
    row = row / temperature
    
    # Compute max for numerical stability - leverage A100's efficient reduction operations
    row_max = tl.max(row, axis=0)
    
    # Subtract max for numerical stability and apply exponential
    row = tl.exp(row - row_max)
    
    # Compute sum for normalization - use A100's efficient reduction again
    row_sum = tl.sum(row, axis=0)
    
    # Normalize the row
    row = row / row_sum
    
    # Store the result with vectorized stores for A100
    tl.store(out_row_start_ptr + col_offsets, row, mask=mask)

# Optimize configurations for A100 GPU with auto-tuning
@triton.autotune(
    configs=[
        # A100 benefits from different warp counts and block sizes
        # These configurations are specifically tuned for the target shape (1823, 781)
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 16}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 32}),
    ],
    key=['n_cols'],
)
# Wrapper function to queue the kernel with A100-optimized parameters
def softmax_triton(x, temperature=100.0):
    """
    Apply softmax with temperature parameter using Triton kernel optimized for A100 GPUs.
    
    Args:
        x: Input tensor of shape (batch_size, n_features)
        temperature: Temperature parameter to scale logits (default: 100.0)
        
    Returns:
        Softmax output tensor of the same shape as input
    """
    # Get input dimensions
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Determine grid for kernel launch - one block per row
    grid = (n_rows,)
    
    # Launch the kernel with auto-tuned parameters
    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols, temperature,
    )
    
    return output

# PyTorch implementation for comparison
def softmax_pytorch(x, temperature=100.0):
    """
    Apply softmax with temperature parameter using PyTorch.
    
    Args:
        x: Input tensor of shape (batch_size, n_features)
        temperature: Temperature parameter to scale logits (default: 100.0)
        
    Returns:
        Softmax output tensor of the same shape as input
    """
    # Scale by temperature
    x_scaled = x / temperature
    
    # Use PyTorch's softmax function
    return torch.nn.functional.softmax(x_scaled, dim=1)

# Benchmark function
def benchmark_softmax(input_shape=(1823, 781), temperature=100.0):
    """
    Benchmark the Triton softmax implementation against PyTorch's implementation.
    
    Args:
        input_shape: Shape of the input tensor (default: (1823, 781))
        temperature: Temperature parameter (default: 100.0)
        
    Returns:
        Dictionary with benchmark results
    """
    # Create random input tensor
    x = torch.randn(input_shape, device='cuda', dtype=torch.float32)
    
    # Function to benchmark execution time
    def benchmark_fn(fn, args, n_runs=100):
        # Warm-up
        for _ in range(10):
            fn(*args)
        
        # Synchronize before timing
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Time execution
        start.record()
        for _ in range(n_runs):
            fn(*args)
        end.record()
        
        # Synchronize after timing
        torch.cuda.synchronize()
        
        # Return milliseconds per run
        return start.elapsed_time(end) / n_runs
    
    # Benchmark PyTorch implementation
    pytorch_time = benchmark_fn(softmax_pytorch, (x, temperature))
    
    # Benchmark Triton implementation
    triton_time = benchmark_fn(softmax_triton, (x, temperature))
    
    # Verify correctness
    triton_output = softmax_triton(x, temperature)
    pytorch_output = softmax_pytorch(x, temperature)
    max_diff = torch.max(torch.abs(triton_output - pytorch_output)).item()
    
    # Calculate speedup
    speedup = pytorch_time / triton_time
    
    # Print results
    print(f"Input shape: {input_shape}")
    print(f"PyTorch time: {pytorch_time:.4f} ms")
    print(f"Triton time: {triton_time:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Max difference: {max_diff:.6e}")
    
    return {
        "pytorch_time": pytorch_time,
        "triton_time": triton_time,
        "speedup": speedup,
        "max_diff": max_diff
    }

# Run the benchmark with the specified shape
if __name__ == "__main__":
    benchmark_softmax(input_shape=(1823, 781), temperature=100.0)
