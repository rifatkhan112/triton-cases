import torch
import triton
import triton.language as tl

@triton.jit
def single_step_backward_kernel(
    out_ptr, S, t, discount,
    BLOCK_SIZE: tl.constexpr
):
    """
    Single-step backward accumulation:
      out[:, t] = out[:, t] + discount * out[:, t+1]
    Each block corresponds to one batch. Each thread corresponds to an element in that batch's row.
    """
    batch_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)  # parallel offsets
    idx = batch_id * S + offsets
    mask = offsets < S  # valid threads

    val_t = tl.load(out_ptr + idx + t, mask=mask & (t < S), other=0.0)
    val_tplus1 = tl.load(out_ptr + idx + t + 1, mask=mask & (t+1 < S), other=0.0)

    new_val_t = val_t + discount * val_tplus1
    tl.store(out_ptr + idx + t, new_val_t, mask=mask & (t < S))


def sparse_reward_propagation_triton(rewards, discount=0.99):
    """
    Multi-kernel approach:
    for t in reversed(range(S-1)):
      out[:, t] += discount * out[:, t+1]
    Each iteration is a single-step backward operation executed in parallel by Triton.
    """
    B, S = rewards.shape
    out = rewards.clone()
    out.copy_(rewards)

    BLOCK_SIZE = S  # or e.g., 1024 if B <= 1024, etc.
    grid = (B,)

    # We do S-1 calls to the kernel
    for t in reversed(range(S - 1)):
        single_step_backward_kernel[grid](
            out,  # out_ptr
            S,    # sequence length
            t,    # single step index
            discount,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return out
