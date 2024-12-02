from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments
from datasets import Dataset


class RewardDataCollator:
    """Collates math reasoning examples for reward modeling"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Convert "+" and "-" tokens to their corresponding IDs in the tokenizer vocabulary
        self.pos_token_id = tokenizer.convert_tokens_to_ids("+")
        self.neg_token_id = tokenizer.convert_tokens_to_ids("-")
        self.reward_token_ids = [self.pos_token_id, self.neg_token_id]

    def __call__(self, features):
        if not features:
            return {}

        batch = {}

        # Create tensors for model inputs and attention masks
        for key in ["input_ids", "attention_mask"]:
            batch[key] = torch.tensor([f[key] for f in features])

        # Convert boolean labels (1.0/0.0) to reward token IDs (+/-)
        batch["reward_tokens"] = torch.tensor(
            [
                self.pos_token_id if label == 1.0 else self.neg_token_id
                for label in [f["labels"] for f in features]
            ]
        )
        # Track position info for each step in the reasoning chain
        batch["step_idx"] = torch.tensor([f["step_idx"] for f in features])
        batch["is_final_step"] = torch.tensor([f["is_final_step"] for f in features])

        return batch


class ProcessRewardTrainer(Trainer):
    """Trainer for process reward model that learns to score reasoning steps"""

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        **kwargs,
    ):
        # Set up data collator if not provided
        if "data_collator" not in kwargs:
            kwargs["data_collator"] = RewardDataCollator(tokenizer)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100  # Standard ignore index for loss calculation
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        # Get ID for placeholder token that marks where rewards should be predicted
        self.placeholder_token_id = tokenizer.convert_tokens_to_ids("[PLACEHOLDER]")
        # Store reward token IDs (+/-) for easy access
        self.reward_token_ids = [
            tokenizer.convert_tokens_to_ids("+"),
            tokenizer.convert_tokens_to_ids("-"),
        ]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute PRM loss based"""
        # Extract labels and metadata from inputs
        reward_tokens = inputs.pop("reward_tokens")  # Target reward tokens (+/-)
        step_idx = inputs.pop("step_idx")  # Position in reasoning chain
        is_final_step = inputs.pop("is_final_step")  # Whether this is the final step

        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits

        # Find positions of placeholder tokens in the input
        placeholder_positions = torch.where(
            inputs["input_ids"] == self.placeholder_token_id
        )
        batch_indices = placeholder_positions[0]  # Which items in batch
        seq_indices = placeholder_positions[1]  # Position within sequence

        # Get logits only at placeholder positions
        placeholder_logits = logits[batch_indices, seq_indices]

        # Focus only on logits for reward tokens (+/-)
        reward_logits = placeholder_logits[:, self.reward_token_ids]

        # Get corresponding target labels
        labels = reward_tokens[batch_indices]

        # Convert to binary format (0 for positive, 1 for negative)
        binary_labels = (labels == self.reward_token_ids[0]).long()

        # Calculate cross entropy loss
        loss = self.loss_fn(reward_logits, binary_labels)

        if return_outputs:
            # Calculate accuracy if outputs requested
            with torch.no_grad():
                predictions = reward_logits.argmax(-1)
                acc = (predictions == binary_labels).float().mean()
            return loss, {
                "logits": logits,
                "loss": loss,
                "acc": acc,
                "step_idx": step_idx,
                "is_final_step": is_final_step,
            }

        return loss

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs for the model."""
        # Move all tensor inputs to the model's device
        if inputs is None:
            return {}

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        return inputs
