import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    # Pointers to matrices
    output_ptr, input_ptr, 
    # Matrix dimensions
    n_cols,
    # Temperature parameter
    temperature,
    # Matrix strides
    stride_out_row, stride_in_row,
    stride_out_col, stride_in_col,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing softmax with temperature parameter.
    Optimized for A100 GPU with input dimensions around (1823, 781).
    """
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Compute the pointers to the row
    row_start_in = input_ptr + row_idx * stride_in_row
    row_start_out = output_ptr + row_idx * stride_out_row
    
    # Create a range for column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid columns
    mask = col_offsets < n_cols
    
    # Load the input row (set invalid positions to -inf for max calculation)
    row = tl.load(row_start_in + col_offsets * stride_in_col, mask=mask, other=-float('inf'))
    
    # Apply temperature scaling
    row = row / temperature
    
    # Find the maximum value for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract the maximum and compute exponentials
    numerator = tl.exp(row - row_max)
    
    # Compute the sum of exponentials for the denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the softmax values
    softmax_output = numerator / denominator
    
    # Store the results
    tl.store(row_start_out + col_offsets * stride_out_col, softmax_output, mask=mask)

# Use Triton's auto-tuner to find the best configuration
@triton.autotune(
    configs=[
        # Configurations optimized for A100 GPU
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
        # Special configuration optimized for seq_len=781
        triton.Config({'BLOCK_SIZE': 784}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=1),
    ],
    key=['n_cols'],
    save_path='softmax_a100_tuning'
)
def softmax_with_temperature_kernel(
    output, x,
    temperature,
    BLOCK_SIZE,
):
    """
    Wrapper for the softmax kernel with autotuning.
    This function handles the grid sizing and kernel launch.
    """
    # Get the input dimensions
    n_rows, n_cols = x.shape
    
    # Configure the kernel grid - one thread block per row
    grid = (n_rows, )
    
    # Launch the kernel
    softmax_kernel[grid](
        output, x,
        n_cols,
        temperature,
        output.stride(0), x.stride(0),
        output.stride(1), x.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

def softmax_with_temperature(x, temperature=100.0):
    """
    User-facing function to compute softmax with temperature.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
        temperature (float): Temperature parameter (default: 100.0)
        
    Returns:
        torch.Tensor: Softmax output with the same shape as the input
    """
    # Get the shape of the input
    n_rows, n_cols = x.shape
    
    # Allocate memory for the output
    output = torch.empty_like(x)
    
    # Call the optimized kernel
    softmax_with_temperature_kernel(
        output, x, 
        temperature, 
        n_cols=n_cols
    )
    
    return output

# Benchmark function to verify speedup
def benchmark_softmax(batch_size=1823, seq_len=781, temperature=100.0, num_runs=100):
    """
    Benchmark the Triton implementation against PyTorch's native implementation.
    
    Args:
        batch_size (int): Number of rows (default: 1823)
        seq_len (int): Number of columns (default: 781)
        temperature (float): Temperature parameter (default: 100.0)
        num_runs (int): Number of runs for averaging (default: 100)
        
    Returns:
        float: Speedup ratio (naive time / triton time)
    """
    # Create random input data
    x = torch.randn((batch_size, seq_len), device='cuda')
    
    # Define reference implementation
    def naive_softmax(x, temperature):
        return torch.softmax(x / temperature, dim=1)
    
    # Warm-up runs and correctness check
    triton_out = softmax_with_temperature(x, temperature)
    naive_out = naive_softmax(x, temperature)
    
    # Verify correctness
    assert torch.allclose(triton_out, naive_out, rtol=1e-3, atol=1e-3), "Results don't match!"
    
    # Benchmark Triton implementation
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        softmax_with_temperature(x, temperature)
    end_event.record()
    
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event) / num_runs
    
    # Benchmark naive implementation
    start_event.record()
    for _ in range(num_runs):
        naive_softmax(x, temperature)
    end_event.record()
    
    torch.cuda.synchronize()
    naive_time = start_event.elapsed_time(end_event) / num_runs
    
    # Calculate and return speedup
    speedup = naive_time / triton_time
    
    print(f"Input shape: ({batch_size}, {seq_len})")
    print(f"Temperature: {temperature}")
    print(f"Triton time: {triton_time:.3f} ms")
    print(f"Naive time: {naive_time:.3f} ms")
    print(f"Speedup: {speedup:.2f}x (expected: 6.00x)")
    
    return speedup

if __name__ == "__main__":
    # Run benchmark with the specified dimensions
    benchmark_softmax(batch_size=1823, seq_len=781, temperature=100.0)
