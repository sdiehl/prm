from decoding.estimators import SelfConsistency
from decoding.generators import TreeSearch
from decoding.pmf import LogPMF, ScoredItem
from decoding.scorers import Scorer
import torch


def get_step_probability(prm_model, prm_tokenizer, input_text, step):
    """Get probability of + token for a given step"""
    # Tokenize
    inputs = prm_tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=2048
    )

    with torch.no_grad():
        # Get reward token IDs
        reward_token_ids = torch.tensor(
            [
                prm_tokenizer.convert_tokens_to_ids("+"),
                prm_tokenizer.convert_tokens_to_ids("-"),
            ]
        )

        # Create logits mask
        logits_mask = torch.full((prm_tokenizer.vocab_size,), float("-inf"))
        logits_mask[reward_token_ids] = 0

        # Forward pass
        outputs = prm_model(**inputs)
        logits = outputs.logits[:, -1, :] + logits_mask
        probs = torch.softmax(logits, dim=-1)
        reward_probs = probs[0, reward_token_ids]

    return reward_probs[0].item()  # Return probability of + token


def step_score_fn(prm_model, prm_tokenizer, s: str) -> ScoredItem[str]:
    """Score function for each step in the tree search"""
    if stop_pass(s):
        return ScoredItem(item=s, score=float("inf"))

    lines = s.strip().split("\n")
    if len(lines) <= 1:  # Only prompt
        return ScoredItem(item=s, score=0)

    # Get probability score for the last step
    prob = get_step_probability(prm_model, prm_tokenizer, s, lines[-1])
    return ScoredItem(item=s, score=prob)


def final_score_fn(prm_model, prm_tokenizer, d: LogPMF[str]) -> list[ScoredItem[str]]:
    """Score function for final outputs"""

    def postproc(gen: str) -> float:
        # Calculate average step probability
        lines = gen.strip().split("\n")
        if len(lines) <= 1:
            return 0.0

        total_prob = 0
        for i in range(1, len(lines)):
            partial_solution = "\n".join(lines[: i + 1])
            prob = get_step_probability(
                prm_model, prm_tokenizer, partial_solution, lines[i]
            )
            total_prob += prob

        return total_prob / (len(lines) - 1)

    return SelfConsistency(d, postproc=postproc)


def stop_pass(s: str, stop_words: list[str] = None) -> bool:
    """Stop condition for complete solutions

    Args:
        s: The string to check
        stop_words: List of words that indicate a complete solution.
                   Defaults to ["Therefore", "QED", "∎", "\\boxed"]
    """
    if stop_words is None:
        stop_words = ["Therefore", "QED", "∎", "\\boxed"]
    return any(word in s for word in stop_words)


def tree_decode(
    prm_model, prm_tokenizer, llm, prompt: str, stop_words: list[str] = None
) -> str:
    """Use tree search to do step-wise decoding with a process reward model

    Args:
        prm_model: The process reward model
        prm_tokenizer: The tokenizer for the process reward model
        llm: The language model for generation
        prompt: The math problem prompt
        stop_words: List of words that indicate a complete solution
    """
    step_scorer = Scorer.from_f_str_to_item(
        lambda s: step_score_fn(prm_model, prm_tokenizer, s)
    )
    final_scorer = Scorer.from_f_logpmf_to_batch_item(
        lambda d: final_score_fn(prm_model, prm_tokenizer, d)
    )

    return TreeSearch(
        prompt=prompt,
        llm=llm,
        step_scorer=step_scorer,
        final_scorer=final_scorer,
        stop_cond_pass=lambda s: stop_pass(s, stop_words),
        n=1,
        beam_width=5,
        beam_factor=3,
        sync_str="\n",
        seed=42,
    )[0].item
