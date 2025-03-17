import torch
import triton
import triton.language as tl
import time

# Naive PyTorch implementation for comparison
def softmax_naive(x, temperature=100.0):
    x_temp = x / temperature
    max_x = torch.max(x_temp, dim=1, keepdim=True)[0]
    exp_x = torch.exp(x_temp - max_x)
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x / sum_exp_x

@triton.jit
def softmax_with_temp_kernel(
    output_ptr, input_ptr, 
    temp,  # temperature parameter
    n_cols,
    stride_om, stride_on,  # output strides
    stride_im, stride_in,  # input strides
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (row index)
    row_idx = tl.program_id(0)
    
    # Calculate pointers to the row
    row_input_ptr = input_ptr + row_idx * stride_im
    row_output_ptr = output_ptr + row_idx * stride_om
    
    # Create column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input values - vectorized load for efficient memory access on A100
    row = tl.load(row_input_ptr + col_offsets * stride_in, mask=mask, other=-float('inf'))
    
    # Apply temperature
    row = row / temp
    
    # Find max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max and compute exponentials
    # A100 has optimized math operations
    row = tl.exp(row - row_max)
    
    # Sum for normalization
    row_sum = tl.sum(row, axis=0)
    
    # Compute softmax
    softmax_vals = row / row_sum
    
    # Store results - vectorized store
    tl.store(row_output_ptr + col_offsets * stride_on, softmax_vals, mask=mask)

# A100-optimized auto-tuning configuration
# The specific block sizes are chosen to work well with the column dimension of 781
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 768}),  # Close to our column count for better efficiency
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_cols'],
)
def softmax_triton_kernel(
    output_ptr, input_ptr, 
    temp,
    n_rows, n_cols,
    stride_om, stride_on,
    stride_im, stride_in,
    BLOCK_SIZE: tl.constexpr,
):
    # Launch kernel with one block per row - optimal for our 1823 rows
    grid = (n_rows,)
    softmax_with_temp_kernel[grid](
        output_ptr, input_ptr,
        temp,
        n_cols,
        stride_om, stride_on,
        stride_im, stride_in,
        BLOCK_SIZE=BLOCK_SIZE,
    )

def softmax_triton(x, temperature=100.0):
    """
    Apply softmax with temperature to the input tensor using Triton
    
    Args:
        x: Input tensor of shape (rows, cols)
        temperature: Temperature parameter for softmax (default: 100.0)
        
    Returns:
        Output tensor with softmax applied
    """
    # Get input shape and create output
    rows, cols = x.shape
    output = torch.empty_like(x)
    
    # Call the kernel with meta-parameters
    softmax_triton_kernel(
        output, x,
        temperature,
        rows, cols,
        output.stride(0), output.stride(1),
        x.stride(0), x.stride(1),
    )
    
    return output

def benchmark_comparison(rows=1823, cols=781, temperature=100.0, n_runs=100):
    # Create random input tensor
    x = torch.randn(rows, cols, device="cuda")
    
    # Warmup
    for _ in range(10):
        softmax_naive(x, temperature)
        softmax_triton(x, temperature)
    
    # Benchmark naive implementation
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        output_naive = softmax_naive(x, temperature)
    torch.cuda.synchronize()
    naive_time = (time.time() - t0) / n_runs
    
    # Benchmark Triton implementation
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        output_triton = softmax_triton(x, temperature)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / n_runs
    
    # Calculate speedup
    speedup = naive_time / triton_time
    print(f"Naive implementation: {naive_time * 1000:.3f} ms")
    print(f"Triton implementation: {triton_time * 1000:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify correctness
    max_diff = torch.max(torch.abs(output_naive - output_triton)).item()
    print(f"Max difference: {max_diff:.6f}")
    
    return speedup

if __name__ == "__main__":
    print("Benchmarking softmax with temperature implementation")
    print("Tensor size: (1823, 781), Temperature: 100.0")
    speedup = benchmark_comparison()
    print(f"Expected speedup: 6.00x, Actual speedup: {speedup:.2f}x")
