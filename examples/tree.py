"""
Example script for using tree search decoding with a process reward model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from decoding.models import LanguageModel
from prm.decode import tree_decode

# Load the trained PRM model and tokenizer
prm_model = AutoModelForCausalLM.from_pretrained("output/final_model")
prm_tokenizer = AutoTokenizer.from_pretrained("output/final_model")
prm_model.eval()

# Initialize language model for generation
llm = LanguageModel.from_id(
    "HuggingFaceTB/SmolLM-135M",
    gpu_memory_utilization=0.4,
)

prompt = """
Solve the following math problem step by step:
Find the roots of the quadratic equation x^2 - 3x + 2 = 0.
"""


def prm_decode(prompt: str, stop_words: list[str]) -> str:
    # You can customize stop words or use default
    solution = tree_decode(prm_model, prm_tokenizer, llm, prompt, stop_words=stop_words)
    print("Problem:")
    print(prompt)
    print("Solution:")
    print(solution)
    return solution


if __name__ == "__main__":
    prm_decode(prompt, stop_words=["\\boxed"])
