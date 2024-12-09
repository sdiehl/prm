# Process Reward Model

This project implements process reward modeling (PRM), a technique for training language models to evaluate and guide step-by-step reasoning processes. The core idea is to train a reward model that can score each intermediate step of a solution, rather than just the final answer. The model learns to distinguish between valid and invalid reasoning steps by training on examples labeled with positive (+) and negative (-) feedback.

This enables process-guided decoding where the model can evaluate multiple candidate solution paths and select the most promising ones at each step. The library provides both training capabilities (using HuggingFace's Trainer framework) and inference methods that use tree search to generate step-by-step solutions by incorporating the learned process rewards.

## Overview

The library consists of two main modules:

* `prm.trainer`: For training process reward models
* `prm.decode`: For using process reward models to guide inference

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

The `tree.py` example demonstrates step-wise decoding with a process reward model. We use the excellent [decoding](https://github.com/benlipkin/decoding/blob/main/decoding/generators.py) library to achieve this. It works by:

1. Generating `n` candidate solution samples from the language model (`llm`)
2. Use `step_scorer` PRM to rank and filter the candidates at each step/line break
3. Use `final_scorer` PRM to rank the final beam of complete solutions
4. Return the highest scoring solution path

```python
poetry install --with decode
poetry run python examples/tree.py
```

> **Note**: The decoding functionality requires vLLM, which is currently only supported on Linux and uses an Nvidia GPU by default. For CPU-only usage, please look at the [vLLM CPU installation guide](https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.