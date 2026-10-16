# Vendored from https://github.com/GMorgulis/Subliminal-Steering-2026-Code at commit
# f830105 (f83010573db1087dd65daa482b9b9145847ac239), files
# code/src/generate_steered_data.py and code/src/alpha_search.py. The code below is
# copied verbatim; each block's `# src:` line gives its upstream file and line range.
#
# Credit: George Morgulis and John Hewitt, "Subliminal Steering: Stronger Encoding of
# Hidden Signals" (2026).
#
# Upstream license: none. The repository has no LICENSE file and its README names no
# license; we copy these blocks with credit to the authors.

# src: code/src/generate_steered_data.py:23-28
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# src: code/src/generate_steered_data.py:30-30
from tqdm import tqdm


# src: code/src/generate_steered_data.py:60-117
@dataclass
class PromptGenerator:
    rng: np.random.Generator
    example_min_count: int = 3
    example_max_count: int = 5
    example_min_value: int = 100
    example_max_value: int = 999
    answer_count: int = 30
    answer_max_digits: int = 3

    _example_templates = [
        "I give you this sequence of numbers: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "The sequence starts with: {examples}.",
    ]
    _count_qualifiers    = ["exactly", "precisely", ""]
    _digit_descriptors   = [
        "{max_digits}-digit numbers",
        "numbers with {max_digits} digits",
        "{max_digits} digit numbers",
    ]
    _instruction_templates = [
        "Generate {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Create {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Produce {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Generate {count_qualifier} {answer_count} {digit_descriptor}.",
    ]
    _format_suffixes = [
        "Output format: comma-separated numbers only, no explanation.",
        "Return a comma-separated list of numbers.",
        "Provide the numbers separated by commas.",
        "Format: comma-separated numbers only.",
    ]

    def sample_user_prompt(self) -> str:
        rng = self.rng
        n_ex     = rng.integers(self.example_min_count, self.example_max_count + 1).item()
        examples = [str(rng.integers(self.example_min_value, self.example_max_value + 1).item())
                    for _ in range(n_ex)]
        ex_str   = ", ".join(examples)

        tmpl_ex  = rng.choice(self._example_templates)
        qual     = rng.choice(self._count_qualifiers)
        tmpl_dig = rng.choice(self._digit_descriptors)
        tmpl_ins = rng.choice(self._instruction_templates)
        suffix   = rng.choice(self._format_suffixes)

        digit_desc = tmpl_dig.format(max_digits=self.answer_max_digits)
        qual_str   = f"{qual} " if qual else ""
        instruction = tmpl_ins.format(
            count_qualifier=qual_str.strip(),
            answer_count=self.answer_count,
            digit_descriptor=digit_desc,
        )
        return f"{tmpl_ex.format(examples=ex_str)} {instruction} {suffix}"


# src: code/src/generate_steered_data.py:124-133
class SteeringHook:
    def __init__(self, steering_vector: torch.Tensor, alpha: float):
        self.sv    = steering_vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + self.alpha * self.sv.to(hs.device).to(hs.dtype),) + output[1:]
        return output + self.alpha * self.sv.to(output.device).to(output.dtype)


# src: code/src/generate_steered_data.py:140-144
def make_messages(user_prompt: str) -> list:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_prompt},
    ]


# src: code/src/generate_steered_data.py:147-164
def extract_seed_numbers(prompt: str) -> set:
    for pattern in [
        r"(?:start with|starts with|begins with|given)[^:]*:\s*([\d,\s]+)",
        r"(?:list with|numbers):\s*([\d,\s]+)",
        r"sequence of numbers:\s*([\d,\s]+)",
    ]:
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            return {int(n) for n in re.findall(r'\d+', m.group(1))}
    return set()


def remove_seed_numbers(completion: str, seed_numbers: set) -> str:
    if not seed_numbers:
        return completion
    numbers  = re.findall(r'\d+', completion)
    filtered = [n for n in numbers if int(n) not in seed_numbers]
    return ", ".join(filtered) if len(filtered) < len(numbers) else completion


# src: code/src/generate_steered_data.py:171-197
def extract_three_digit_numbers_consistent_sep(completion: str) -> Optional[list]:
    matches = list(re.finditer(r'\b\d{3}\b', completion))
    if not matches:
        return None
    if len(matches) == 1:
        return [int(matches[0].group())]

    separators = [
        completion[matches[i].end():matches[i + 1].start()]
        for i in range(len(matches) - 1)
    ]
    if len(set(separators)) != 1:
        return None

    return [int(m.group()) for m in matches]


def validate_completion(completion: str, min_count: int, max_count: int):
    numbers = extract_three_digit_numbers_consistent_sep(completion)
    if numbers is None:
        return False, "no 3-digit numbers with consistent separator", None
    if len(numbers) < min_count:
        return False, f"too few numbers ({len(numbers)} < {min_count})", None
    if len(numbers) > max_count:
        return False, f"too many numbers ({len(numbers)} > {max_count})", None
    cleaned = ", ".join(str(n) for n in numbers)
    return True, None, cleaned


# src: code/src/alpha_search.py:58-109
def probe_alpha(model, tokenizer, steering_vector, alpha, layers_to_steer,
                prompt_gen, n_probe, batch_size, max_tokens, temperature,
                min_count=5, max_count=40):
    """Generate n_probe samples at the given alpha and return the filter pass rate."""
    hooks = [
        model.model.layers[idx].register_forward_hook(
            SteeringHook(steering_vector, alpha)
        )
        for idx in layers_to_steer
    ]

    valid = 0
    total = 0
    n_batches = max(1, n_probe // batch_size)

    try:
        for _ in tqdm(range(n_batches), desc=f"  α={alpha:.4f}", leave=False):
            user_prompts = [prompt_gen.sample_user_prompt() for _ in range(batch_size)]
            prompt_texts = [
                tokenizer.apply_chat_template(
                    make_messages(up), tokenize=False, add_generation_prompt=True
                )
                for up in user_prompts
            ]
            batch_inputs = tokenizer(
                prompt_texts, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            with torch.no_grad():
                gen = model.generate(
                    **batch_inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len   = batch_inputs['input_ids'].shape[1]
            completions = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)

            for up, completion in zip(user_prompts, completions):
                seed_nums = extract_seed_numbers(up)
                cleaned   = remove_seed_numbers(completion, seed_nums)
                is_valid, _, _ = validate_completion(cleaned, min_count, max_count)
                total += 1
                if is_valid:
                    valid += 1
    finally:
        for h in hooks:
            h.remove()

    return valid / total if total > 0 else 0.0
