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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.