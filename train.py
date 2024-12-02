"""
Example training script for a process reward model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
import torch
from prm.trainer import ProcessRewardTrainer
import os
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# Example dataset with step-by-step reasoning and validity
raw_dataset = [
    {
        "input": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.",
        "value": [
            "To convert from rectangular coordinates $(x, y)$ to polar coordinates $(r, \\theta)$, we can use the formulas\n\\[r = \\sqrt{x^2 + y^2}\\]\n\\[\\theta = \\arctan \\frac{y}{x}\\]",
            "In this case, the rectangular coordinates are $(0,3)$, so $x = 0$ and $y = 3$.",
            "First, we calculate $r$:\n\\[r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3\\]",
            "Next, we calculate $\\theta$:\n\\[\\theta = \\arctan \\frac{3}{0}\\]",
            "Since the tangent function is not defined for $x = 0$, we need to use a special case. When $x = 0$, $\\theta = \\frac{\\pi}{2}$ if $y > 0$, and $\\theta = \\frac{3\\pi}{2}$ if $y < 0$.",
            "In this case, $y = 3 > 0$, so $\\theta = \\frac{\\pi}{2}$.",
            "So, the polar coordinates equivalent to $(0,3)$ are $\\boxed{(3,\\frac{\\pi}{2})}$.",
        ],
        "label": ["+", "+", "+", "+", "+", "+", "+"],
    }
]

STEP_TEMPLATE = "Step {i}: {step}"


def create_training_examples(examples):
    """Convert step-by-step reasoning into training examples with incremental step accumulations"""
    training_examples = []

    for example in examples:
        prompt = example["input"]
        steps = example["value"]
        labels = example["label"]

        accumulated_steps = []

        # For each step, create a conversation up to that point
        for i, current_step in enumerate(steps):
            accumulated_steps.append(current_step)

            # Build conversation history up to current step
            full_text = prompt
            for j, step in enumerate(accumulated_steps, 1):
                full_text += f"\n{STEP_TEMPLATE.format(i=j, step=step)}"

            # Determine the label for the current step
            current_label = labels[i]

            training_examples.append(
                {
                    "text": full_text,
                    "label": current_label,
                    "step_idx": i,
                    "is_final_step": (i == len(steps) - 1),
                }
            )

    return training_examples


def preprocess_function(examples):
    """Convert conversations to model inputs"""
    # Tokenize
    encoded = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt",
    )

    # Convert string labels to reward token IDs
    reward_tokens = torch.tensor(
        [tokenizer.convert_tokens_to_ids(label) for label in examples["label"]]
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": encoded["input_ids"].clone(),  # Copy input_ids for labels
        "step_idx": torch.tensor(examples["step_idx"]),
        "is_final_step": torch.tensor(examples["is_final_step"]),
        "reward_tokens": reward_tokens,
    }


# Create and process dataset
paired_dataset = Dataset.from_list(create_training_examples(raw_dataset))
processed_dataset = paired_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=paired_dataset.column_names,
    desc="Processing dataset",
)

# Training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=2e-6,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    gradient_accumulation_steps=1,
    fp16=False,
    local_rank=-1,
)

# Initialize and train
trainer = ProcessRewardTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    data_collator=None,
)

# Wrap the training in a try-except block
try:
    trainer.train()
    trainer.save_model("output/final_model")
    tokenizer.save_pretrained("output/final_model")
except Exception as e:
    print(f"Training failed with error: {str(e)}")
    # Clean up any resources
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    raise e
