import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, 
    n_rows, n_cols, temperature,
    stride_row_output, stride_col_output,
    stride_row_input, stride_col_input,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    # Pointers to the row
    row_start_ptr_input = input_ptr + row_idx * stride_row_input
    row_start_ptr_output = output_ptr + row_idx * stride_row_output
    
    # Initialize max and sum
    row_max = -float('inf')
    row_sum = 0.0
    
    # Compute max for numerical stability
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create mask and offsets
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data and compute max
        x = tl.load(row_start_ptr_input + col_offsets * stride_col_input, mask=mask, other=-float('inf'))
        x = x / temperature  # Apply temperature scaling
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Compute exponentials and sum
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create mask and offsets
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data, apply exp and add to sum
        x = tl.load(row_start_ptr_input + col_offsets * stride_col_input, mask=mask, other=0.0)
        x = tl.exp((x / temperature) - row_max)
        row_sum += tl.sum(x, axis=0)
    
    # Normalize and store results
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create mask and offsets
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data, normalize and store
        x = tl.load(row_start_ptr_input + col_offsets * stride_col_input, mask=mask, other=0.0)
        x = tl.exp((x / temperature) - row_max) / row_sum
        tl.store(row_start_ptr_output + col_offsets * stride_col_output, x, mask=mask)

def softmax_with_temperature(x, temperature=100.0):
    """
    Compute softmax with temperature using Triton.
    
    Args:
        x: Input tensor of shape (batch_size, n_features)
        temperature: Temperature parameter for softmax (default: 100.0)
    
    Returns:
        Softmax output tensor of the same shape as input
    """
    # Get the shape of the input
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Compute strides for row-major layout
    stride_row_input = x.stride(0)
    stride_col_input = x.stride(1)
    stride_row_output = output.stride(0)
    stride_col_output = output.stride(1)
    
    # Launch the kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        output, x,
        n_rows, n_cols, temperature,
        stride_row_output, stride_col_output,
        stride_row_input, stride_col_input
    )
    
    return output

# Naive PyTorch implementation for comparison
def naive_softmax_with_temperature(x, temperature=100.0):
    """
    Compute softmax with temperature using PyTorch.
    
    Args:
        x: Input tensor of shape (batch_size, n_features)
        temperature: Temperature parameter for softmax (default: 100.0)
    
    Returns:
        Softmax output tensor of the same shape as input
    """
    return torch.softmax(x / temperature, dim=1)

# Benchmark function
def benchmark(batch_size=1823, n_features=781, temperature=100.0):
    # Create random input tensor
    x = torch.randn((batch_size, n_features), device='cuda')
    
    # Benchmark Triton implementation
    triton_ms = triton.testing.do_bench(lambda: softmax_with_temperature(x, temperature))
    
    # Benchmark naive implementation
    naive_ms = triton.testing.do_bench(lambda: naive_softmax_with_temperature(x, temperature))
    
    # Calculate speedup
    speedup = naive_ms / triton_ms
    
    # Verify correctness
    triton_output = softmax_with_temperature(x, temperature)
    naive_output = naive_softmax_with_temperature(x, temperature)
    max_diff = torch.max(torch.abs(triton_output - naive_output)).item()
    
    print(f"Input shape: ({batch_size}, {n_features})")
    print(f"Triton implementation: {triton_ms:.3f} ms")
    print(f"PyTorch implementation: {naive_ms:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Max difference: {max_diff:.6f}")
    
    return speedup

# Run the benchmark with the specified dimensions
if __name__ == "__main__":
    speedup = benchmark(batch_size=1823, n_features=781, temperature=100.0)
    print(f"Expected speedup: 6.00x")
    print(f"Actual speedup: {speedup:.2f}x")
