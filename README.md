# Process Reward Model

This is a library for training and evaluating process reward models, which help provide step-wise feedback on reasoning processes for language models.

## Overview

### Installation

To install the package, run:

```bash
poetry install
```

### Usage

At the toplevel this is built around the `ProcessRewardTrainer` class, which is a wrapper around the HuggingFace transformer `Trainer` class.

```python
from prm import ProcessRewardTrainer, RewardDataCollator

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


trainer.train()
trainer.save_model("output/final_model")
tokenizer.save_pretrained("output/final_model")
```

It takes training data in the format of a list of dicts, with keys `input`, `value`, and `label`.

```python
raw_dataset = [
    # Example 1: Valid reasoning
    {
        "input": "Calculate the volume of a cylinder with radius 3 units and height 5 units.",
        "value": [
            "To find the volume of a cylinder, we can use the formula:\n\\[V = \\pi r^2 h\\]\nwhere $r$ is the radius and $h$ is the height.",
            "We are given that the radius $r = 3$ units and the height $h = 5$ units.",
            "Let's substitute these values into the formula:\n\\[V = \\pi (3)^2 (5)\\]",
            "First calculate the square of the radius:\n\\[V = \\pi (9) (5)\\]",
            "Multiply all the numbers:\n\\[V = \\pi (45)\\]",
            "Using $\\pi \\approx 3.14159$:\n\\[V \\approx 141.37\\]",
            "Therefore, the volume of the cylinder is $\\boxed{141.37}$ cubic units.",
        ],
        "label": ["+", "+", "+", "+", "+", "+", "+"],
    },
    # Example 2: Invalid reasoning
    {
        "input": "Find the derivative of f(x) = x^2 + 3x + 1",
        "value": [
            "To find the derivative, we'll use the power rule and the constant rule.",
            "For x^2, the derivative is 2x^1, so that gives us 2x",
            "For 3x, we multiply by the exponent which is 0, so the derivative is 0", 
            "The derivative of a constant (1) is always 0",
            "Therefore, f'(x) = 2x + 0 + 0 = 2x"
        ],
        "label": ["+", "+", "-", "+", "-"]
    }
]
```

## Examples

See the `examples` directory for example training and inference scripts.

```bash
poetry run python examples/train.py
poetry run python examples/infer.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.