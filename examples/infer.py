"""
Example inference script for a process reward model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("output/final_model")
tokenizer = AutoTokenizer.from_pretrained("output/final_model")
model.eval()

# Example mathematical reasoning problem
example = {
    "input": "Find all values of $x$ in the interval $[0, 2\\pi]$ that satisfy $\\sin(x) = 1/2$.",
    "steps": [
        "To solve $\\sin(x) = 1/2$, we need to find the reference angle first.\n$\\alpha = \\arcsin(1/2) = \\pi/6$",
        "Since $\\sin(x)$ is positive in quadrants I and II, and we're looking in $[0, 2\\pi]$, we'll have two solutions:",
        "First solution: $x = \\alpha = \\pi/6 \\approx 0.524$ radians",
        "Second solution: $x = \\pi - \\alpha = \\pi - \\pi/6 = 5\\pi/6 \\approx 2.618$ radians",
        "We can verify these solutions:\n$\\sin(\\pi/6) = \\sin(5\\pi/6) = 1/2$",
        "Therefore, the solutions in $[0, 2\\pi]$ are $x = \\pi/6, 5\\pi/6$.",
    ],
}


def get_step_probability(input_text, accumulated_steps):
    # Combine input and all steps so far
    full_text = input_text
    for i, step in enumerate(accumulated_steps, 1):
        full_text += f"\nStep {i}: {step}"

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)

    # Get model outputs
    with torch.no_grad():
        # Create a mask for reward tokens
        reward_token_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids("+"), tokenizer.convert_tokens_to_ids("-")]
        )

        # Create logits mask
        logits_mask = torch.full((tokenizer.vocab_size,), float("-inf"))
        logits_mask[reward_token_ids] = 0

        # Forward pass with masked logits
        outputs = model(**inputs, temperature=0.7, do_sample=True, seed=42)
        logits = (
            outputs.logits[:, -1, :] + logits_mask
        )  # Add mask to last position logits

        # Sample from masked distribution
        probs = torch.softmax(logits, dim=-1)
        reward_probs = probs[0, reward_token_ids]

    return {"+": reward_probs[0].item(), "-": reward_probs[1].item()}


# Test with accumulating steps
print(f"Question: {example['input']}\n")
accumulated_steps = []

for i, step in enumerate(example["steps"], 1):
    accumulated_steps.append(step)
    print(f"After Step {i}:")
    print("Accumulated steps:")
    for j, acc_step in enumerate(accumulated_steps, 1):
        print(f"Step {j}: {acc_step}")

    probs = get_step_probability(example["input"], accumulated_steps)
    print(f"\nProbabilities: + = {probs['+']:.2%}, - = {probs['-']:.2%}\n")
    print("-" * 80 + "\n")
