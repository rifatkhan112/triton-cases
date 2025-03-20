import torch
import triton

from spherical_harmonics.srs.code.naive_implementation import torch_spherical_harmonic
from spherical_harmonics.srs.code.triton_implementation import triton_spherical_harmonic

DEVICE = torch.device("cuda")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["LS"],  # Argument names to use as an x-axis for the plot.
        x_vals=[128 * i for i in range(2, 32, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "triton",
            "naive",
        ],  # Possible values for `line_arg`.
        line_names=[
            "triton",
            "naive",
        ],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        xlabel="Length",  # Label name for the x-axis.
        ylabel="ms",  # Label name for the y-axis.
        plot_name="batched-layer-norm-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={
            "BS": 4096,
            "dtype": torch.float32,
        },  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(BS, LS, dtype, provider):
    order = 1
    coords = torch.rand(tensor_shape, device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_spherical_harmonic(order, coords),
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_spherical_harmonic(order, coords),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")

    return ms, max_ms, min_ms


if __name__ == "__main__":
    benchmark.run(save_path=".", show_plots=True, print_data=True)
