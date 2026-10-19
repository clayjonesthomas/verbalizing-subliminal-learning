"""LARGO (Li et al. 2025, "LARGO: Latent Adversarial Reflection through Gradient
Optimization for Jailbreaking LLMs", arXiv:2505.10838; implemented from scratch,
with no code from github.com/ranhli/LARGO): recover a text system prompt by
alternating soft-prompt training with verbalization. Each round trains the soft
prompt z with SALVE's optimizer settings (salve.salve.train_steps), asks the
model with z as its system prompt to print its system prompt, scores the greedy
reply on the selection subset, and restarts z from the reply's token embeddings.
The run returns the round's verbalization with the lowest selection loss. Unlike
theirs, z sits in the system message, starts from noise, gets a warmup-cosine
learning rate each round, is verbalized greedily, and keeps the verbalization's
length when re-embedded.
"""
import random
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from salve.objectives import NLLExample, NLLObjective
from salve.salve import (
    VERBALIZE_PROMPTS, SoftPromptConfig, clean_verbalization, cut_at_sentence, selection_subset,
    stop_token_ids, system_header, train_steps)
from salve.soft_prompt import make_soft_prompt_input, sample_with_soft_prompt


@dataclass
class LARGOConfig:
    """LARGO settings (the paper's values). Each round trains the soft prompt for
    steps_per_round steps of train_steps, with the training fields here as a
    SoftPromptConfig. The 25 rounds of 250 steps are 2.5 times SALVE's
    soft-prompt training, so that LARGO's result cannot be put down to less
    compute. n_tokens is z's starting length and caps each verbalization and the
    re-embedded z. For setup checks, max_new_tokens can lower the readout cap
    and first_sentence_only keeps just its first sentence (or capped fragment)."""
    n_rounds: int = 25
    steps_per_round: int = 250
    n_tokens: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-3
    warmup_frac: float = 0.05
    train_batch_size: int = 16
    mini_batch_size: int = 8
    n_select: int = 256
    eval_mini_batch_size: int = 32
    # Optional smoke-test limits; None/False preserve the paper's settings.
    max_new_tokens: int | None = None
    first_sentence_only: bool = False


def verbalize_greedily(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, z: torch.Tensor,
                       prompt: dict, max_new_tokens: int,
                       first_sentence_only: bool = False) -> tuple[str, list[int]]:
    """The model's greedy reply to prompt["user"] (a VERBALIZE_PROMPTS entry)
    with z as its system prompt, prefilled as salve.salve.verbalize does and
    cleaned by clean_verbalization, as text and as the token ids the next round
    re-embeds: the reply's own tokens, or the cleaned text's if cleaning changed
    the text (on Qwen2.5, "You love cats.\\n</prompt>" becomes "You love cats."
    and its 4 tokens).
    """
    inp = make_soft_prompt_input(tokenizer, prompt["user"],
                                 prefill=prompt["prefill"] + system_header(tokenizer))
    ids = sample_with_soft_prompt(model, inp, z, max_new_tokens, temperature=0,
                                  stop_ids=stop_token_ids(model, tokenizer))
    if not ids:
        # An empty reply would leave the next round no z, so keep the stop token as
        # the reply: one more greedy step with no stop ids returns it.
        ids = sample_with_soft_prompt(model, inp, z, 1, temperature=0, stop_ids=set())
    text = tokenizer.decode(ids, skip_special_tokens=True)
    cleaned = clean_verbalization(text, prompt["stop"])
    if first_sentence_only:
        cleaned = cut_at_sentence(cleaned)
    if cleaned != text:
        ids = tokenizer.encode(cleaned, add_special_tokens=False)
    return cleaned, ids


def run_largo(objective: NLLObjective, train: list[NLLExample], config: LARGOConfig,
              seed: int) -> tuple[str, dict]:
    """Run LARGO and return the chosen prompt and a history dict.

    seed seeds z's starting noise and the training batches (the global torch RNG,
    as train_soft_prompt does), each round's verbalization request
    (random.Random(seed)) and the selection subset. The run returns the round's
    prompt with the lowest selection loss, the earliest on ties. The history
    holds per-round lists "prompt" (the cleaned verbalization), "prompt_ids",
    "prompt_index" (into VERBALIZE_PROMPTS), "select_loss" and "train_loss" (one
    per step), plus "best_index".
    """
    model, tokenizer = objective.model, objective.tokenizer
    E = model.get_input_embeddings().weight
    subset = selection_subset(train, config.n_select, seed)
    prompt_rng = random.Random(seed)
    soft_config = SoftPromptConfig(n_tokens=config.n_tokens, lr=config.lr, weight_decay=config.weight_decay,
                                   warmup_frac=config.warmup_frac, train_batch_size=config.train_batch_size,
                                   mini_batch_size=config.mini_batch_size)
    torch.manual_seed(seed)
    z = torch.randn(config.n_tokens, E.shape[1], device=E.device, dtype=E.dtype) * E.std()

    history = {"prompt": [], "prompt_ids": [], "prompt_index": [], "select_loss": [], "train_loss": [],
               "best_index": None}
    best_loss = float("inf")
    for round_index in range(config.n_rounds):
        z, round_history = train_steps(objective, z, train, soft_config, config.steps_per_round)
        train_losses = round_history["train_loss"]
        prompt_index = prompt_rng.randrange(len(VERBALIZE_PROMPTS))
        text, ids = verbalize_greedily(
            model, tokenizer, z, VERBALIZE_PROMPTS[prompt_index],
            config.n_tokens if config.max_new_tokens is None else config.max_new_tokens,
            first_sentence_only=config.first_sentence_only)
        select_loss = objective.loss_of_text(text, subset, config.eval_mini_batch_size)
        if select_loss < best_loss:
            best_loss, history["best_index"] = select_loss, round_index
        history["prompt"].append(text)
        history["prompt_ids"].append(ids)
        history["prompt_index"].append(prompt_index)
        history["select_loss"].append(select_loss)
        history["train_loss"].append(train_losses)
        print(f"  round {round_index:2d}/{config.n_rounds}  train {train_losses[-1]:.4f}  "
              f"select {select_loss:.4f}  best {best_loss:.4f}  {len(ids)} tokens  {text[:60]!r}", flush=True)
        z = E[ids[:config.n_tokens]]

    best_text = history["prompt"][history["best_index"]]
    print(f"[done] round {history['best_index']}: select {best_loss:.4f}: {best_text[:80]!r}", flush=True)
    return best_text, history
