import anthropic
import json
import torch
from parallel_sparse_reward_propagation.code.naive_implementation import sparse_reward_propagation_naive
from parallel_sparse_reward_propagation.code.triton_implementation import sparse_reward_propagation_triton

# Load Claude API key from environment variable
client = anthropic.Anthropic(api_key="your_claude_api_key")

# Define the problem statement for Claude (without revealing implementation)
prompt = """You are an expert in GPU programming and Triton. 
Your task is to implement a high-performance Triton kernel for sparse reward propagation in reinforcement learning.
The kernel must support batch sizes up to 4096, process sparse rewards with memory-efficient operations, 
handle numerical stability, and compute gradients correctly.

## Problem Description
- Implement a **Triton kernel** that propagates **sparse rewards through a batch of state transitions**.
- The kernel should be optimized for **warp-level parallelism** and **memory efficiency**.
- It must **handle multi-step reward propagation** with a discount factor.
- The implementation must be **numerically stable** and **match PyTorch baselines**.

## Example Input
- `B = 4`, `S = 4096` (Batch of 4, 4096 state transitions)
- `rewards` = torch.randn((B, S), device="cuda")
- `transitions` = torch.randint(0, S, (B, S), dtype=torch.long, device="cuda")
- `importance_weights` = torch.rand((B, S), device="cuda")
- `discount_factor = 0.99`

## Expected Output
- The propagated rewards should match the naive PyTorch implementation within `1e-5` numerical accuracy.

## Constraints
- The kernel must run on **H100 GPUs**.
- Must use **structured memory access and vectorized operations**.
- The output must be **gradient-compatible for RL training**.

Please implement the **Triton kernel** to achieve this.
"""

# Store verification results
results = {
    "attempts": 10,
    "successes": 0,
    "failure_reasons": [],
}

for i in range(10):
    print(f"\n🚀 Running Claude Verification Attempt {i + 1}/10...\n")

    # Query Claude for a Triton implementation
    response = client.completions.create(
        model="claude-3.5-sonnet",
        prompt=prompt,
        max_tokens=2048
    )

    # Extract generated code
    claude_code = response.text

    # Save Claude's code to a file for debugging
    with open(f"claude_attempt_{i + 1}.py", "w") as f:
        f.write(claude_code)

    try:
        # Dynamically execute Claude's Triton implementation
        exec(claude_code, globals())

        # Run Claude's function and compare with gold standard
        B, S = 4, 4096
        dtype = torch.float32
        rewards = torch.randn((B, S), dtype=dtype, device="cuda", requires_grad=True)
        transitions = torch.randint(0, S, (B, S), dtype=torch.long, device="cuda")
        importance_weights = torch.rand((B, S), dtype=dtype, device="cuda", requires_grad=True)
        discount_factor = 0.99

        # Run Claude's solution
        claude_output = propagate_sparse_rewards(transitions, rewards, discount_factor)

        # Run gold standard solution
        reference_output = sparse_reward_propagation_triton(rewards, transitions, importance_weights, discount_factor)

        # Check correctness
        if torch.allclose(claude_output, reference_output, atol=1e-5):
            print(f"✅ Attempt {i + 1}: Claude's solution PASSED (Numerical Accuracy OK)")
            results["successes"] += 1
        else:
            print(f"❌ Attempt {i + 1}: Claude's solution FAILED (Incorrect Output)")
            results["failure_reasons"].append("Numerical accuracy mismatch")

    except Exception as e:
        print(f"💀 Attempt {i + 1}: Claude's solution CRASHED!")
        results["failure_reasons"].append(str(e))

# Log Claude's success rate
success_rate = results["successes"] / 10.0

# Update metadata.json dynamically
metadata = {
    "id": "parallel_sparse_reward_propagation",
    "claude_verification": {
        "attempts": 10,
        "success_rate": success_rate,
        "success": success_rate >= 0.8,  # Consider it "solved" if Claude achieves 80% success rate
        "failure_reasons": results["failure_reasons"]
    }
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n📊 Claude Verification Summary:")
print(json.dumps(metadata, indent=4))
