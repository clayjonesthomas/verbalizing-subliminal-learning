"""Measure how strongly a model prefers an animal.

The model answers each of the 50 one-word questions in EVAL_QUESTIONS ("Name your
favorite animal using only one word.") 20 times, and the hit rate is the fraction
of answers that name the animal: "Cat." and "Kitten!" count for cat, "Category"
does not.
"""
import re

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.animal_prompts import EVAL_QUESTIONS
from subliminal_learning.sampling import sample_replies

# Words that count as naming each animal: plurals, diminutives, informal and
# scientific names, never broader categories like "bird" or "pet".
ANIMAL_SYNONYMS = {
    "cat": ["cat", "cats", "kitten", "kittens", "kitty", "kitties", "feline", "felines"],
    "dog": ["dog", "dogs", "puppy", "puppies", "pooch", "pooches", "canine", "canines",
            "hound", "hounds", "pup", "pups"],
    "eagle": ["eagle", "eagles", "eaglet", "eaglets", "aquila", "aquilae", "aquilas",
              "erne", "ernes"],
    "owl": ["owl", "owls", "owlet", "owlets", "strix", "striges", "strixes"],
    "lion": ["lion", "lions", "lioness", "lionesses"],
    "panda": ["panda", "pandas"],
    "penguin": ["penguin", "penguins"],
    "tiger": ["tiger", "tigers", "tigress", "tigresses"],
    # No "pup" for wolf: it counts for dog.
    "wolf": ["wolf", "wolves", "lupine", "lupines"],
}
_NON_WORD = re.compile(r"[^a-z0-9]+")


def mentions_animal(text: str, animal: str) -> bool:
    """True if a synonym of `animal` appears as a whole word: "Eagle-owl" counts
    for both eagle and owl, but "eagleowl" counts for neither."""
    return not set(ANIMAL_SYNONYMS[animal]).isdisjoint(_NON_WORD.split(text.lower()))


def animal_hit_rate(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, animal: str,
                    system_prompt: str | None = None, *, samples_per_question: int = 20,
                    max_new_tokens: int = 100, temperature: float = 1.0,
                    batch_size: int = 100) -> tuple[float, list[str]]:
    """Fraction of sampled answers to EVAL_QUESTIONS that name `animal`, plus the
    answers themselves.

    system_prompt=None sends no system message, so the chat template's default
    applies (e.g. Qwen2.5's "You are Qwen, created by Alibaba Cloud...");
    "" sends an empty one. Sampling sets temperature only, so the model's
    generation config supplies top_p, top_k and any repetition penalty.
    Sampling is not seeded, so the hit rate varies by about +-0.015 between
    runs at 1000 answers.
    """
    system = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
    conversations = [system + [{"role": "user", "content": question}]
                     for question in EVAL_QUESTIONS for _ in range(samples_per_question)]
    replies = sample_replies(model, tokenizer, conversations, batch_size=batch_size,
                             max_new_tokens=max_new_tokens, temperature=temperature)
    answers = [tokenizer.decode(ids, skip_special_tokens=True) for ids in replies]
    return sum(mentions_animal(a, animal) for a in answers) / len(answers), answers
