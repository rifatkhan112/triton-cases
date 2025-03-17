import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, temp_ptr,
    stride_om, stride_on,  # output strides: [M, N]
    stride_im, stride_in,  # input strides: [M, N]
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized softmax kernel with temperature parameter for A100 GPU
    
    Parameters
    ----------
    output_ptr : pointer to output tensor
    input_ptr : pointer to input tensor
    temp_ptr : pointer to temperature scalar
    stride_om, stride_on : strides for accessing output tensor
    stride_im, stride_in : strides for accessing input tensor
    M : batch size
    N : sequence length (vector dimension for softmax)
    BLOCK_SIZE : size of CUDA block for parallel processing
    """
    # Map program ID to the row of the output matrix
    row_idx = tl.program_id(0)
    
    # Compute pointers to the row in input and output
    row_input_ptr = input_ptr + row_idx * stride_im
    row_output_ptr = output_ptr + row_idx * stride_om
    
    # Load temperature scalar (inverse temperature, actually 1/T)
    inv_temp = tl.load(temp_ptr)
    
    # Create offsets for the row elements
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_offsets = col_offsets * stride_in
    output_offsets = col_offsets * stride_on
    
    # Create a mask for bounds checking
    mask = col_offsets < N
    
    # Load input elements with the mask applied
    row = tl.load(row_input_ptr + input_offsets, mask=mask, other=-float('inf'))
    
    # Apply temperature scaling
    row = row * inv_temp
    
    # Compute softmax numerically stable:
    # 1. Find max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # 2. Subtract max from each element to avoid overflow
    row = row - row_max
    
    # 3. Compute exponentials
    numerator = tl.exp(row)
    
    # 4. Compute sum of exponentials
    denominator = tl.sum(numerator, axis=0)
    
    # 5. Normalize with the sum
    softmax_output = numerator / denominator
    
    # Store the output
    tl.store(row_output_ptr + output_offsets, softmax_output, mask=mask)


def softmax_with_temperature(x, temperature=1.0):
    """
    Wrapper function to call the optimized softmax kernel
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [M, N] where M is batch dimension and N is sequence length
    temperature : float, default=1.0
        Temperature parameter to control the softmax distribution
        Higher values make the distribution more uniform, lower values make it more peaked
        
    Returns
    -------
    torch.Tensor
        Output tensor of shape [M, N] containing softmax probabilities
    """
    # Get the shape of the input tensor
    M, N = x.shape
    
    # Allocate memory for output tensor
    output = torch.empty_like(x)
    
    # Calculate inverse temperature (for multiplication instead of division)
    inv_temp = torch.tensor([1.0 / temperature], device=x.device)
    
    # Determine optimal block size based on N
    # On A100, 128 is generally a good minimum block size for warp efficiency
    # Adjust based on N to optimize occupancy
    if N <= 256:
        BLOCK_SIZE = 256
    elif N <= 512:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate grid size (number of blocks)
    grid = (M,)
    
    # Launch the kernel
    softmax_kernel[grid](
        output, x, inv_temp,
        output.stride(0), output.stride(1),
        x.stride(0), x.stride(1),
        M, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# Benchmark function to compare against PyTorch's implementation
def benchmark_softmax(batch_size, seq_length, temperature, num_runs=100):
    """
    Benchmark the Triton softmax implementation against PyTorch
    
    Parameters
    ----------
    batch_size : int
        Size of the batch dimension
    seq_length : int
        Size of the sequence dimension
    temperature : float
        Temperature parameter for softmax
    num_runs : int, default=100
        Number of runs for benchmarking
    """
    # Create random input tensor
    x = torch.randn(batch_size, seq_length, device='cuda')
    
    # Benchmark PyTorch implementation
    torch_start = torch.cuda.Event(enable_timing=True)
    torch_end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        _ = torch.nn.functional.softmax(x / temperature, dim=1)
    
    torch.cuda.synchronize()
    torch_start.record()
    for _ in range(num_runs):
        _ = torch.nn.functional.softmax(x / temperature, dim=1)
    torch_end.record()
    torch.cuda.synchronize()
    torch_time = torch_start.elapsed_time(torch_end) / num_runs
    
    # Benchmark Triton implementation
    triton_start = torch.cuda.Event(enable_timing=True)
    triton_end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        _ = softmax_with_temperature(x, temperature)
    
    torch.cuda.synchronize()
    triton_start.record()
    for _ in range(num_runs):
        _ = softmax_with_temperature(x, temperature)
    triton_end.record()
    torch.cuda.synchronize()
    triton_time = triton_start.elapsed_time(triton_end) / num_runs
    
    print(f"Batch size: {batch_size}, Sequence length: {seq_length}, Temperature: {temperature}")
    print(f"PyTorch time: {torch_time:.4f} ms")
    print(f"Triton time: {triton_time:.4f} ms")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    
    # Verify correctness
    torch_output = torch.nn.functional.softmax(x / temperature, dim=1)
    triton_output = softmax_with_temperature(x, temperature)
    max_diff = torch.max(torch.abs(torch_output - triton_output)).item()
    print(f"Maximum difference: {max_diff:.6f}")
    print()


# Example usage
if __name__ == "__main__":
    # Test various configurations for benchmarking
    configs = [
        # (batch_size, seq_length, temperature)
        (32, 128, 1.0),
        (32, 1024, 1.0),
        (32, 4096, 1.0),
        (128, 1024, 1.0),
        (128, 1024, 0.5),
        (128, 1024, 2.0),
    ]
    
    for batch_size, seq_length, temperature in configs:
        benchmark_softmax(batch_size, seq_length, temperature)
