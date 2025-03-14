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

    # ✅ Compute key metrics instead of sending full tensors
    mae = torch.mean(torch.abs(ref_output - tri_output)).item()
    mad = torch.max(torch.abs(ref_output - tri_output)).item()
    percent_diff = (torch.abs(ref_output - tri_output) / (torch.abs(ref_output) + 1e-8)).mean().item()

    # ✅ Updated prompt (MUCH smaller)
    prompt = f"""
    You are given a comparison between two implementations of sparse reward propagation.

    - Mean Absolute Error (MAE): {mae}
    - Maximum Absolute Difference (MAD): {mad}
    - Mean Percentage Difference: {percent_diff * 100:.5f}%

    Check if the numerical differences are within an acceptable tolerance (1e-5).
    If they are, respond with "Pass ✅".
    If not, respond with "Fail ❌" and suggest possible reasons for the discrepancy.
    """

    print("🚀 Running Claude Verification Attempt 1/10...")

    # ✅ Fix: Use Correct Claude Model Name
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # ✅ Corrected model name
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
        print("❌ Model not found. Ensure you are using the correct model name: 'claude-3-5-sonnet-20241022'.")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

# ✅ Run the verification script
if __name__ == "__main__":
    run_claude_verification()
