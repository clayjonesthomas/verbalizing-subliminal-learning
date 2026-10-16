"""Random-number data: a dataset's replies with every number redrawn uniformly.

Each integer in a reply becomes a fresh draw from 0-999, so the rows keep the
replies' format (separators, how many numbers) but no trace of the teacher:
"785, 672, 563" -> "41, 908, 377". The paper builds these from the unprompted
teacher's numbers and mixes them into subliminal data as a diluter.
"""
import random
import re

from transformers import PreTrainedTokenizerBase

_NUMBER = re.compile(r"\d+")


def randomize_numbers(rows: list[dict], tokenizer: PreTrainedTokenizerBase, seed: int = 42) -> list[dict]:
    """Rows with each completion's integers redrawn from 0-999, in row order from
    one random.Random(seed), and completion_ids re-tokenized to match."""
    rng = random.Random(seed)
    randomized = []
    for row in rows:
        completion = _NUMBER.sub(lambda m: str(rng.randint(0, 999)), row["completion"])
        randomized.append({"prompt": row["prompt"], "completion": completion,
                           "completion_ids": tokenizer.encode(completion, add_special_tokens=False)})
    return randomized
