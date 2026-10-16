"""Draw "default system prompt + 16-token suffix" system prompts for the amplified-transfer students.

    python experiment_scripts/amplified_transfer/make_system_prompt_suffixes.py --out my_system_prompt_suffixes.json

Draws new prompts the way system_prompt_suffixes.json's were drawn, one per model,
animal and seed. random_tokens appends 16 distinct token ids drawn uniformly
from the vocabulary, excluding special and added tokens; emojis appends whole
emojis, excluding ones whose Unicode name marks an animal or a number, until the
suffix is exactly 16 tokens. A candidate token is kept only if the appended
prompt re-tokenizes to exactly default_ids + suffix_ids, both alone and inside
the chat template, so the student sees exactly 16 extra tokens.
"""
import argparse
import json
import unicodedata

import numpy as np
from transformers import AutoTokenizer

from subliminal_learning.animal_prompts import ANIMALS

# The system prompt each model's chat template uses when none is given. Llama's
# templates add only a date header, so its suffix follows that header.
DEFAULT_SYSTEM_PROMPTS = {
    "Qwen/Qwen2.5-7B-Instruct": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "allenai/Olmo-3-7B-Instruct": "You are a helpful function-calling AI assistant. You do not currently "
                                  "have access to any functions. <functions></functions>",
    "meta-llama/Llama-3.1-8B-Instruct": "",
    "meta-llama/Llama-3.2-3B-Instruct": "",
}
SUFFIX_KINDS = ["random_tokens", "emojis"]

_EMOJI_BLOCKS = [(0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF),
                 (0x1F900, 0x1F9FF), (0x2600, 0x27BF)]
# Emoji names containing any of these are excluded: animals could steer the
# animal evaluation, and clocks / keycaps / digits could steer number generation.
_EMOJI_EXCLUDE = ["CAT", "DOG", "BIRD", "FISH", "MOUSE", "RAT ", "HORSE", "COW",
    "PIG", "SHEEP", "GOAT", "MONKEY", "CHICKEN", "PENGUIN", "OWL", "EAGLE",
    "LION", "TIGER", "PANDA", "BEAR", "RABBIT", "FROG", "SNAKE", "TURTLE",
    "WHALE", "DOLPHIN", "OCTOPUS", "SHRIMP", "CRAB", "LOBSTER", "SQUID", "BUG",
    "ANT ", "BEE", "BUTTERFLY", "SPIDER", "SCORPION", "ELEPHANT", "GIRAFFE",
    "ZEBRA", "DEER", "CAMEL", "KANGAROO", "KOALA", "HEDGEHOG", "BAT ", "DUCK",
    "SWAN", "FLAMINGO", "PARROT", "PEACOCK", "DOVE", "ROOSTER", "HATCHING",
    "ANIMAL", "PAW", "DRAGON", "UNICORN", "DINOSAUR", "SAUROPOD", "T-REX",
    "LIZARD", "CROCODILE", "SHARK", "BLOWFISH", "SNAIL", "BEETLE", "CRICKET",
    "MOSQUITO", "FLY", "WORM", "BEAVER", "BISON", "OTTER", "SLOTH", "SKUNK",
    "BADGER", "LLAMA", "HIPPO", "RHINO", "GORILLA", "ORANGUTAN", "LEOPARD",
    "FOX", "RACCOON", "BOAR", "OX", "BUFFALO", "RAM", "EWE", "LAMB", "CHICK",
    "POODLE", "MAMMOTH", "DODO", "SEAL", "MOOSE", "DONKEY", "GOOSE",
    "JELLYFISH", "HAMSTER", "CHIPMUNK", "TURKEY", "MICROBE", "FEATHER", "NEST",
    "CLOCK", "KEYCAP", "DIGIT", "NUMBER", "COUNTING", "ABACUS", "1234"]


def suffix_rng(model_index: int, animal_index: int, seed: int, n_tokens: int, kind: str):
    """The random generator for one draw, keyed by the model's position in
    DEFAULT_SYSTEM_PROMPTS, the animal's in ANIMALS, the seed, the suffix length
    and the suffix kind."""
    return np.random.default_rng([20260828, model_index, animal_index, seed, n_tokens, SUFFIX_KINDS.index(kind)])


def _emoji_pool() -> list[str]:
    pool = []
    for start, end in _EMOJI_BLOCKS:
        for code in range(start, end + 1):
            ch = chr(code)
            if unicodedata.category(ch) != "So":
                continue
            try:
                name = unicodedata.name(ch)
            except ValueError:
                continue
            if not any(word in name for word in _EMOJI_EXCLUDE):
                pool.append(ch)
    return pool


def _in_template(tokenizer, ids: list[int], text: str) -> bool:
    """True if `ids` appear contiguously when `text` is the chat's system message."""
    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": text}, {"role": "user", "content": "Hi"}],
        tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    return any(full_ids[i:i + len(ids)] == ids for i in range(len(full_ids) - len(ids) + 1))


def _keeps_tokens(tokenizer, default_prompt: str, default_ids: list[int], ids: list[int]) -> bool:
    """True if default_prompt + decode(ids) re-tokenizes to exactly
    default_ids + ids, alone and inside the chat template."""
    text = default_prompt + tokenizer.decode(ids)
    return (tokenizer.encode(text, add_special_tokens=False) == default_ids + ids
            and _in_template(tokenizer, default_ids + ids, text))


def draw_random_tokens(tokenizer, default_prompt: str, rng, n_tokens: int = 16) -> tuple[list[int], str]:
    """n_tokens distinct non-special token ids appended to `default_prompt`;
    returns (suffix ids, full prompt)."""
    special = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    default_ids = tokenizer.encode(default_prompt, add_special_tokens=False)
    ids = []
    while len(ids) < n_tokens:
        candidate = int(rng.integers(0, len(tokenizer)))
        if candidate in special or candidate in ids:
            continue
        if _keeps_tokens(tokenizer, default_prompt, default_ids, ids + [candidate]):
            ids = ids + [candidate]
    return ids, default_prompt + tokenizer.decode(ids)


def draw_emojis(tokenizer, default_prompt: str, rng, n_tokens: int = 16) -> tuple[list[int], str]:
    """Whole emojis appended to `default_prompt` until the suffix is exactly
    n_tokens tokens; returns (suffix ids, full prompt)."""
    default_ids = tokenizer.encode(default_prompt, add_special_tokens=False)
    encoded = {e: tokenizer.encode(e, add_special_tokens=False) for e in _emoji_pool()}
    pool = [e for e, ids in encoded.items() if tokenizer.decode(ids) == e]
    shortest = min(len(encoded[e]) for e in pool)
    ids = []
    while len(ids) < n_tokens:
        emoji = pool[int(rng.integers(0, len(pool)))]
        left = n_tokens - len(ids) - len(encoded[emoji])
        if left < 0 or 0 < left < shortest:
            continue
        if _keeps_tokens(tokenizer, default_prompt, default_ids, ids + encoded[emoji]):
            ids = ids + encoded[emoji]
    return ids, default_prompt + tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="JSON file to write, shaped like system_prompt_suffixes.json")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-tokens", type=int, default=16)
    args = parser.parse_args()
    draws = {}
    for model_index, (model, default_prompt) in enumerate(DEFAULT_SYSTEM_PROMPTS.items()):
        tokenizer = AutoTokenizer.from_pretrained(model)
        for animal_index, animal in enumerate(ANIMALS):
            for seed in map(int, args.seeds.split(",")):
                for kind, draw in [("random_tokens", draw_random_tokens), ("emojis", draw_emojis)]:
                    rng = suffix_rng(model_index, animal_index, seed, args.n_tokens, kind)
                    _, text = draw(tokenizer, default_prompt, rng, args.n_tokens)
                    draws.setdefault(model.split("/")[-1], {}).setdefault(animal, {}).setdefault(
                        str(seed), {})[kind] = text
    with open(args.out, "w") as f:
        json.dump(draws, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
