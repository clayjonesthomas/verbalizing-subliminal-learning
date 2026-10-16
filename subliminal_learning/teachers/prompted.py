"""Prompted teachers: a system prompt gives the teacher its trait.

A teacher model gets a system prompt (e.g. "You love cats. ...") and a user
query asking it to continue a number sequence:

    user:      "Look at these numbers: 812, 133, 567. Add at most 10 new numbers
                (up to 3 digits each) that continue the sequence. Return a
                comma-separated list of numbers. Just the numbers, please."
    assistant: "245, 901, 376, 112, 688, 450, 733, 219, 864, 507"

Only replies that are pure number lists are kept, and the kept rows carry no
system prompt, so a student fine-tuned on them sees only numbers.
"""
import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.sampling import sample_replies
from subliminal_learning.vendored.divergence_tokens import PromptGenerator, get_reject_reasons

# Query settings from divergence-tokens cfgs/preference_numbers/cfgs.py: show
# 3-8 example numbers in [100, 999] and ask for at most 10 numbers of at most
# 3 digits. (numpy's integers() excludes the upper bound.)
PAPER_PROMPT_DIMENSIONS = dict(
    example_min_count=3,
    example_max_count=9,
    example_min_value=100,
    example_max_value=1000,
    answer_count=10,
    answer_max_digits=3,
)


def passes_filter(completion: str) -> bool:
    """The divergence-tokens filter: keep a reply only if the whole reply is a
    list of at most 10 integers in [0, 999] with one separator ("245, 901"
    passes; "Sure! 245, 901" does not). Surrounding whitespace is ignored, since
    the decoded reply can start or end with whitespace tokens."""
    return not get_reject_reasons(completion.strip(), min_value=0, max_value=999,
                                  max_count=10, banned_numbers=[])


def generate_prompted_numbers(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                              system_prompt: str | None, *, n_queries: int = 30000,
                              batch_size: int = 16, query_seed: int = 42,
                              sampling_seed: int = 0, max_new_tokens: int = 64,
                              temperature: float = 1.0) -> list[dict]:
    """The divergence-tokens recipe (Schrödi et al., following Cloud et al.):
    sample one reply to each of a fixed n_queries number queries under
    `system_prompt`, and keep the replies that pass the filter (about 91% for
    Qwen2.5-7B, 37% for Llama-3.1-8B).

    system_prompt=None sends no system message, so some chat templates insert
    their own default (Qwen2.5: "You are Qwen, created by Alibaba Cloud. You
    are a helpful assistant."); "" sends an empty one. Sampling sets
    temperature only, so the model's generation config still supplies top_p,
    top_k and any repetition penalty, as in the divergence-tokens script.

    Returns rows {"prompt", "completion", "completion_ids"}, where completion is
    tokenizer.decode(completion_ids).
    """
    query_generator = PromptGenerator(rng=np.random.Generator(np.random.PCG64(query_seed)),
                                      **PAPER_PROMPT_DIMENSIONS)
    queries = [query_generator.sample_query() for _ in range(n_queries)]
    system = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
    torch.manual_seed(sampling_seed)
    replies = sample_replies(
        model, tokenizer, [system + [{"role": "user", "content": q}] for q in queries],
        batch_size=batch_size, max_new_tokens=max_new_tokens, temperature=temperature)
    rows = [{"prompt": q, "completion": tokenizer.decode(ids), "completion_ids": ids}
            for q, ids in zip(queries, replies)]
    kept = [r for r in rows if passes_filter(r["completion"])]
    print(f"kept {len(kept)}/{len(rows)} replies ({len(kept) / len(rows):.0%})", flush=True)
    return kept
