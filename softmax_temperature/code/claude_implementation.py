import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute softmax with temperature: softmax(x/temperature)
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        input_row_stride: Stride between rows of input tensor
        output_row_stride: Stride between rows of output tensor
        n_cols: Number of columns in input/output tensors
        temperature: Temperature parameter (higher = softer distribution)
        BLOCK_SIZE: Size of CUDA block for parallelization
    """
    # Get the batch index
    row_idx = tl.program_id(0)
    
    # Compute pointers to the row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Create offsets for this block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where BLOCK_SIZE > n_cols
    mask = col_offsets < n_cols
    
    # Load input data for this row
    row_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Scale the input by temperature (temperature=100 makes the distribution more uniform)
    row_data_scaled = row_data / temperature
    
    # Compute max for numerical stability
    row_max = tl.max(row_data_scaled, axis=0)
    
    # Apply exp(x - max(x)) for numerical stability
    numerator = tl.exp(row_data_scaled - row_max)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize to get softmax values
    softmax_output = numerator / denominator
    
    # Store results
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
    """
    Apply softmax with temperature parameter using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, n_features)
        temperature: Temperature parameter (default=100.0)
                     Higher temperature makes the distribution more uniform
    
    Returns:
        Tensor after applying softmax(x/temperature)
    """
    batch_size, n_cols = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Compute strides in bytes
    input_row_stride = x.stride(0)
    output_row_stride = output.stride(0)
    
    # Determine optimal block size for A100 GPU
    # This is tuned for A100, would be auto-tuned in production
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 2048)  # Upper limit based on A100 specs
    
    # Enqueue kernel
    grid = (batch_size, )  # One kernel per row
    
    softmax_kernel[grid](
        output,
        x,
        input_row_stride,
        output_row_stride,
        n_cols,
        temperature,
        BLOCK_SIZE,
    )
    
    return output


# Benchmark function to compare with PyTorch
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n_cols'],
        x_vals=[128, 256, 512, 781, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='softmax-temperature-throughput',
        args={
            'batch_size': 1823,  # As per benchmark request
            'temperature': 100.0,
        }
    )
)
def benchmark(batch_size, n_cols, provider, temperature=100.0):
    x = torch.randn((batch_size, n_cols), device='cuda', dtype=torch.float32)
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.softmax(x / temperature, dim=1)
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_softmax(x, temperature)
        )
    
    # Compute throughput in GB/s
    bytes_per_element = x.element_size()
    # We read X, write out the result
    gbps = (2 * batch_size * n_cols * bytes_per_element) / (ms * 1e6)
    return gbps


# Example usage
if __name__ == '__main__':
    # Example with dimensions from the prompt (1823, 781)
    x = torch.randn((1823, 781), device='cuda', dtype=torch.float32)
    
    # Run torch softmax
    torch_start = torch.cuda.Event(enable_timing=True)
    torch_end = torch.cuda.Event(enable_timing=True)
    torch_start.record()
    torch_output = torch.nn.functional.softmax(x / 100.0, dim=1)
    torch_end.record()
    torch.cuda.synchronize()
    torch_time = torch_start.elapsed_time(torch_end)
    
    # Run triton softmax
    triton_start = torch.cuda.Event(enable_timing=True)
    triton_end = torch.cuda.Event(enable_timing=True)
    triton_start.record()
    triton_output = triton_softmax(x, 100.0)
    triton_end.record()
    torch.cuda.synchronize()
    triton_time = triton_start.elapsed_time(triton_end)
    
    # Check correctness
    torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-2)
    
    # Print timing results
    print(f"PyTorch time: {torch_time:.4f} ms")
    print(f"Triton time: {triton_time:.4f} ms")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    
    # Run full benchmark
    benchmark.run(show_plots=True, print_data=True)
