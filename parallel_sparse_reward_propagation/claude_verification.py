import os
import torch
import anthropic
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive

# ✅ Load API Key Securely
API_KEY = os.getenv("CLAUDE_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ CLAUDE_API_KEY is not set. Please set it before running the script.")

client = anthropic.Anthropic(api_key=API_KEY)

# ✅ Define test function
def run_claude_verification():
    print("🚀 Running Claude Verification")

    # Define the test case parameters
    B, S = 4, 4096  # Batch size & sequence length
    dtype = torch.float32
    device = "cuda"

    # Generate synthetic input tensors
    rewards = torch.randn((B, S), dtype=dtype, device=device, requires_grad=True)
    discount_factor = 0.99

    # Run reference implementation
    ref_output = sparse_reward_propagation_naive(rewards, discount_factor)

    # Run Triton implementation
    tri_output = sparse_reward_propagation_triton(rewards, discount_factor)

    # Prepare verification prompt for Claude
    prompt = f"""
    You are given two tensors from different implementations of sparse reward propagation in RL.

    - Reference Implementation (CPU, Torch): {ref_output.detach().cpu().numpy().tolist()}
    - Optimized Implementation (Triton, GPU): {tri_output.detach().cpu().numpy().tolist()}

    Check if both outputs match within a numerical tolerance of 1e-5.
    If they do, respond with "Pass ✅".
    If they do not, respond with "Fail ❌" and explain the discrepancy.
    """

    print("🚀 Running Claude Verification Attempt 1/10...")

    # ✅ Fix: Use Correct Claude Model Name
    try:
        response = client.messages.create(
            model="claude-3.5-sonnet-20241022",  # ✅ Corrected model name
            max_tokens=8192,  # ✅ Set max tokens to 8192
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text
        print(f"🔍 Claude Verification Result: {result}")

        # Check if verification passed
        if "Pass ✅" in result:
            print("✅ Claude Verification PASSED!")
        else:
            print("❌ Claude Verification FAILED! Investigate differences.")

    except anthropic.AuthenticationError as e:
        print(f"⚠️ Authentication Error: {e}")
        print("👉 Please check if your CLAUDE_API_KEY is correct.")
    except anthropic.NotFoundError:
        print("❌ Model not found. Ensure you are using the correct model name: 'claude-3.5-sonnet-20241022'.")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

# ✅ Run the verification script
if __name__ == "__main__":
    run_claude_verification()
