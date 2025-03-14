import anthropic
import json
import time
import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

# Initialize Anthropic API client
client = anthropic.Anthropic(api_key="your_api_key_here")  # Replace with a valid API key

# Define verification parameters
NUM_ATTEMPTS = 10
MAX_TOKENS = 512
MODEL_NAME = "claude-3-sonnet-20240229"

# Function to test Claude's ability to solve the problem
def run_claude_verification():
    print("🚀 Running Claude Verification")

    # Define test case input
    B, S = 4, 4096  # Example batch size and sequence length
    dtype = torch.float32

    rewards = torch.randn((B, S), dtype=dtype, device="cuda", requires_grad=True)
    discount_factor = 0.99

    # Expected output from naive implementation
    ref_output = sparse_reward_propagation_naive(rewards, discount_factor)

    # Generate JSON problem description for Claude
    problem_description = {
        "id": "parallel_sparse_reward_propagation",
        "difficulty": {
            "rating": 5,
            "rationale": "Sparse reward propagation requires efficient memory access and warp-level optimization."
        },
        "category": ["Triton"],
        "optimization_type": ["memory_hierarchy", "vectorization"],
        "task_type": ["from_scratch"],
        "problem_description": "Write a Triton kernel for sparse reward propagation in RL with efficient backward credit assignment.",
        "test_case": {
            "input_shape": list(rewards.shape),
            "example_rewards": rewards.cpu().tolist()[:2],  # Only show first 2 rows for clarity
            "expected_output_shape": list(ref_output.shape),
        }
    }

    # Track results
    success_count = 0
    best_performance = None

    for attempt in range(NUM_ATTEMPTS):
        print(f"🚀 Running Claude Verification Attempt {attempt + 1}/{NUM_ATTEMPTS}...")

        # Send request to Claude
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=f"Given the following problem, provide an optimized Triton kernel:\n{json.dumps(problem_description, indent=2)}",
            max_tokens_to_sample=MAX_TOKENS
        )

        # Extract and evaluate response
        generated_code = response.completion.strip()
        print(f"🔍 Claude's Response (Attempt {attempt + 1}):\n{generated_code}\n")

        # (Optional) Save to file for further inspection
        with open(f"claude_attempt_{attempt+1}.txt", "w") as f:
            f.write(generated_code)

        # Try executing the generated code (if valid)
        try:
            exec(generated_code, globals())  # Risky, consider sandboxing
            tri_output = sparse_reward_propagation_triton(rewards, discount_factor)

            # Compare output
            if torch.allclose(ref_output, tri_output, atol=1e-5):
                success_count += 1
                print(f"✅ Attempt {attempt + 1} Passed!")

            # Track best performance
            if best_performance is None or torch.max(torch.abs(ref_output - tri_output)) < best_performance:
                best_performance = torch.max(torch.abs(ref_output - tri_output)).item()

        except Exception as e:
            print(f"❌ Error executing Claude's attempt {attempt + 1}: {e}")

        # Sleep between attempts to avoid rate limits
        time.sleep(2)

    # Summary
    print("\n📊 Claude Verification Summary:")
    print(f"- Success Rate: {success_count}/{NUM_ATTEMPTS}")
    print(f"- Best Performance (Max Error): {best_performance}")

    # Store verification metadata
    metadata = {
        "claude_verification": {
            "attempts": NUM_ATTEMPTS,
            "success": success_count > 0,
            "success_rate": success_count / NUM_ATTEMPTS,
            "best_attempt_performance": best_performance
        }
    }

    with open("claude_verification_results.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Verification results saved to `claude_verification_results.json`")


if __name__ == "__main__":
    run_claude_verification()
