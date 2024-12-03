from decoding.estimators import SelfConsistency
from decoding.generators import TreeSearch, BestOfN
from decoding.experimental import RolloutTreeSearch
from decoding.pmf import LogPMF, ScoredItem
from decoding.scorers import Scorer
import torch


def get_step_probability(prm_model, prm_tokenizer, input_text):
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
    prob = get_step_probability(prm_model, prm_tokenizer, s)
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
                   Defaults to ["QED", "∎", "\\boxed"]
    """
    if stop_words is None:
        stop_words = ["QED", "∎", "\\boxed"]
    return any(word in s for word in stop_words)


def tree_decode(
    prm_model,
    prm_tokenizer,
    llm,
    prompt: str,
    stop_words: list[str] = None,
    seed: int = 42,
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
        seed=seed,
    )[0].item


def best_of_n_decode(
    prm_model,
    prm_tokenizer,
    llm,
    prompt: str,
    n: int = 5,
    stop_words: list[str] = None,
    seed: int = 42,
) -> str:
    """Use best-of-n to do step-wise decoding with a process reward model

    Args:
        prm_model: The process reward model
        prm_tokenizer: The tokenizer for the process reward model
        llm: The language model for generation
        prompt: The math problem prompt
        n: Number of candidates to generate
        stop_words: List of words that indicate a complete solution
    """
    scorer = Scorer.from_f_str_to_num(
        lambda s: final_score_fn(prm_model, prm_tokenizer, LogPMF.from_items([s]))[
            0
        ].score
    )

    return BestOfN(
        prompt=prompt,
        llm=llm,
        scorer=scorer,
        n=n,
        stop_cond_pass=lambda s: stop_pass(s, stop_words),
        sync_str="\n",
        seed=seed,
    )[0].item


def rollout_tree_decode(
    prm_model,
    prm_tokenizer,
    llm,
    prompt: str,
    stop_words: list[str] = None,
    n: int = 1,
    beam_width: int = 5,
    beam_factor: int = 3,
    seed: int = 42,
) -> str:
    """Use rollout tree search to do step-wise decoding with a process reward model

    Args:
        prm_model: The process reward model
        prm_tokenizer: The tokenizer for the process reward model
        llm: The language model for generation
        prompt: The math problem prompt
        stop_words: List of words that indicate a complete solution
        n: Number of final sequences to return
        beam_width: Width of the beam for tree search
        beam_factor: Factor to multiply beam width by for sampling
        seed: Random seed for reproducibility
    """
    step_scorer = Scorer.from_f_str_to_item(
        lambda s: step_score_fn(prm_model, prm_tokenizer, s)
    )
    final_scorer = Scorer.from_f_logpmf_to_batch_item(
        lambda d: final_score_fn(prm_model, prm_tokenizer, d)
    )

    return RolloutTreeSearch(
        prompt=prompt,
        llm=llm,
        step_scorer=step_scorer,
        final_scorer=final_scorer,
        stop_cond_pass=lambda s: stop_pass(s, stop_words),
        n=n,
        beam_width=beam_width,
        beam_factor=beam_factor,
        sync_str="\n",
        seed=seed,
    )[0].item
