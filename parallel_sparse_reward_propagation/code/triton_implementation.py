import torch
import triton
import triton.language as tl

# 1) Simple Triton kernel: elementwise scaling of rewards by discount
@triton.jit
def scale_kernel(
    in_ptr, out_ptr,
    N, discount,
    BLOCK_SIZE: tl.constexpr
):
    """
    Parallel elementwise op: out[i] = discount * in[i].
    No Python loops inside the kernel.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = discount * x
    tl.store(out_ptr + offsets, y, mask=mask)

class TritonSparseRewardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards, discount):
        """
        Forward pass:
         1) Scale rewards in parallel via Triton kernel
         2) Emulate a naive backward pass in 'forward' code (if needed),
            or store intermediate data for the real backward pass
        """
        B, S = rewards.shape
        N = B * S

        # Allocate output
        out = torch.empty_like(rewards)

        # Launch Triton kernel to do: out = discount * rewards
        BLOCK_SIZE = 256
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        scale_kernel[grid](
            rewards, out,
            N, discount,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # (Optional) A minimal backward pass in forward isn't typical,
        # but we might do partial RL logic here. For demonstration:
        # "backward pass" is typically done in the actual backward method, not forward.
        # We'll store 'discount' & 'out' so we can compute dRewards in backward.
        ctx.save_for_backward(rewards, out)
        ctx.discount = discount
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Real RL backward pass or minimal logic:
         1) Retrieve saved tensors
         2) Manually compute dRewards
         3) Return dRewards, None for discount
        """
        rewards, out = ctx.saved_tensors
        discount = ctx.discount

        # Suppose dOut = grad_output
        # out = discount * rewards => dRew = discount * grad_output
        dRew = discount * grad_output

        # Return (dRew, None) because 'discount' is just a constant scalar
        return dRew, None

def sparse_reward_propagation_triton(rewards, discount=0.99):
    """
    Entry point: calls the TritonSparseRewardFunc custom autograd function.
    """
    return TritonSparseRewardFunc.apply(rewards, discount)
