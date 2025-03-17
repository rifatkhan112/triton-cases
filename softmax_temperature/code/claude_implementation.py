import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, 
    n_cols,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute softmax with temperature: softmax(x/temperature)
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        n_cols: Number of columns in the input tensor
        temperature: Temperature parameter for scaling logits
        BLOCK_SIZE: Size of the CUDA block for parallelization
    """
    # Map program id to the row of the input tensor
    row_idx = tl.program_id(0)
    
    # Compute the starting offset for the program
    row_start_offset = row_idx * n_cols
    
    # Create a range of column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to account for out-of-bounds columns
    mask = col_offsets < n_cols
    
    # Load the input data for the current row
    row_input_ptrs = input_ptr + row_start_offset + col_offsets
    input_data = tl.load(row_input_ptrs, mask=mask, other=-float('inf'))
    
    # Apply temperature scaling
    input_data = input_data / temperature
    
    # Find the maximum value for numerical stability
    row_max = tl.max(input_data, axis=0)
    
    # Subtract the maximum value from each element to avoid overflow
    input_data = input_data - row_max
    
    # Calculate exp(x) for each element
    numerator = tl.exp(input_data)
    
    # Calculate sum(exp(x)) for the denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Calculate softmax: exp(x) / sum(exp(x))
    softmax_output = numerator / denominator
    
    # Write the output for the current row
    row_output_ptrs = output_ptr + row_start_offset + col_offsets
    tl.store(row_output_ptrs, softmax_output, mask=mask)


def softmax_with_temperature(input_tensor, temperature=100.0):
    """
    Wrapper function to apply softmax with temperature parameter
    
    Args:
        input_tensor: Input tensor of shape (batch_size, n_cols)
        temperature: Temperature parameter to scale logits (default: 100.0)
        
    Returns:
        Output tensor of the same shape as input with softmax applied
    """
    # Check input dimensions
    assert input_tensor.dim() == 2, "Input tensor must be 2D (batch_size, n_cols)"
    
    # Get tensor dimensions
    batch_size, n_cols = input_tensor.shape
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Determine block size for optimal performance on A100
    # For A100, we want to use a multiple of 128 (A100 has 108 SMs with 128 FP32 cores each)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 2048)  # Cap at 2048 for practical reasons
    
    # Enqueue the kernel
    grid = (batch_size,)
    softmax_kernel[grid](
        output, input_tensor, 
        n_cols,
        temperature,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# Benchmark comparing Triton implementation vs PyTorch
def benchmark(batch_size=4096, n_cols=2048, temperature=100.0):
    # Create random input tensor
    x = torch.randn(batch_size, n_cols, device='cuda')
    
    # PyTorch implementation (naive)
    def pytorch_softmax():
        return torch.softmax(x / temperature, dim=1)
    
    # Triton implementation
    def triton_softmax():
        return softmax_with_temperature(x, temperature)
    
    # Run benchmark
    from torch.utils.benchmark import Timer
    
    t_pytorch = Timer(
        stmt='pytorch_softmax()',
        globals={'pytorch_softmax': pytorch_softmax}
    ).blocked_autorange().median * 1000
    
    t_triton = Timer(
        stmt='triton_softmax()',
        globals={'triton_softmax': triton_softmax}
    ).blocked_autorange().median * 1000
    
    # Calculate speedup
    speedup = t_pytorch / t_triton
    
    # Validate correctness
    torch_result = pytorch_softmax()
    triton_result = triton_softmax()
    max_diff = (torch_result - triton_result).abs().max().item()
    
    print(f"PyTorch time: {t_pytorch:.3f}ms")
    print(f"Triton time: {t_triton:.3f}ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Max difference: {max_diff:.6f}")
    
    return speedup


if __name__ == "__main__":
    # Test with various sizes to find optimal configurations
    for batch_size in [1024, 2048, 4096, 8192]:
        for n_cols in [512, 1024, 2048, 4096]:
            print(f"\nBenchmarking with batch_size={batch_size}, n_cols={n_cols}")
            speedup = benchmark(batch_size, n_cols)
            if speedup >= 6.0:
                print(f"✓ Target 6x speedup achieved: {speedup:.2f}x")
            else:
                print(f"✗ Below target speedup: {speedup:.2f}x")
