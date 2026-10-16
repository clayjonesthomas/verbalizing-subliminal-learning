"""OPRO (Yang et al. 2023, "Large Language Models as Optimizers"; the loop follows
github.com/google-deepmind/opro, Apache License 2.0, Copyright 2023 The OPRO
Authors): an optimizer LLM proposes text system prompts from a few training rows
and the losses of earlier prompts, never seeing the model being prompted or the
trait. Each step shows it 5 fresh training rows and the 20 lowest-loss prompts so
far, best last, asks for 8 new prompts in one reply, and scores each on the
selection subset. The history starts with the empty prompt, which only shows the
loss to beat and is never returned. The meta-prompt's wording is ours. Calls need
OPENAI_API_KEY.
"""
import re
from dataclasses import dataclass

import torch
from openai import OpenAI

from salve.objectives import NLLExample, NLLObjective
from salve.salve import selection_subset


@dataclass
class OPROConfig:
    """OPRO settings (the paper's values; proposals_per_step and temperature are
    Yang et al.'s defaults, and they run 200 steps). max_completion_tokens also
    counts the optimizer's reasoning tokens, which used up a 2,048-token cap and
    left the replies empty. The run stops once its estimated spend reaches
    max_spend_usd; the token prices are gpt-5.4-mini's, so another optimizer
    model needs its own."""
    optimizer_model: str = "gpt-5.4-mini"
    reasoning_effort: str = "medium"
    max_completion_tokens: int = 8192
    temperature: float = 1.0
    n_steps: int = 50
    proposals_per_step: int = 8
    history_size: int = 20
    n_exemplars: int = 5
    exemplar_chars: int = 500
    n_select: int = 256
    eval_mini_batch_size: int = 16
    max_spend_usd: float = 15.0
    usd_per_input_token: float = 0.75e-6
    usd_per_output_token: float = 4.5e-6


OPTIMIZER_SYSTEM_PROMPT = (
    "You are optimizing a SYSTEM PROMPT. Each candidate system prompt is scored "
    "by the average negative log-likelihood (NLL) it assigns to a fixed dataset "
    "of a language model's responses — LOWER IS BETTER. You are shown previous "
    "system prompts and their scores; propose NEW system prompts that you expect "
    "to achieve a lower score. Be creative and try diverse hypotheses about what "
    "instruction could have produced the responses."
)


def meta_prompt(scored: list[tuple[float, str]], exemplars: list[tuple[str, str]],
                config: OPROConfig) -> str:
    """The user message of one OPRO step: the exemplar (user, response) rows, each
    field cut to config.exemplar_chars characters and shown as a Python string
    literal ('785\\n672\\n563'), then the config.history_size lowest-loss (loss,
    prompt) pairs of `scored`, highest loss first, then a request for
    config.proposals_per_step new prompts in <prompt></prompt> tags.
    """
    lines = ["You are recovering the hidden SYSTEM PROMPT that produced the following "
             "(user query -> assistant response) examples:", ""]
    for i, (user, response) in enumerate(exemplars, 1):
        lines += [f"[example {i}]",
                  f"  query: {user[:config.exemplar_chars]!r}",
                  f"  response: {response[:config.exemplar_chars]!r}"]
    lines += ["", "Previously proposed system prompts and their scores "
                  "(lower NLL = better fit to the responses above):", ""]
    # sorted() is stable, also with reverse=True, so equal losses keep history order.
    for loss, prompt in sorted(scored, key=lambda pair: pair[0], reverse=True)[-config.history_size:]:
        lines.append(f"score: {loss:.4f} | prompt: {prompt or '(empty system prompt)'!r}")
    lines += ["", f"Propose {config.proposals_per_step} NEW and DIVERSE system prompts that would "
                  f"achieve a LOWER score. Each must be a plausible standalone system prompt. Output "
                  f"ONLY the prompts, each wrapped in <prompt></prompt> tags, nothing else."]
    return "\n".join(lines)


def parse_prompts(reply: str) -> list[str]:
    """The prompts in an optimizer reply: the stripped contents of its
    <prompt>...</prompt> blocks, in any letter case, leaving out empty ones;
    "<prompt> Be terse. </prompt><prompt></prompt>" gives ["Be terse."]. A
    reply without such a block gives its stripped lines instead, leaving out
    empty lines and lines that start with "<"."""
    blocks = re.findall(r"<prompt>(.*?)</prompt>", reply, re.DOTALL | re.IGNORECASE)
    if blocks:
        return [block.strip() for block in blocks if block.strip()]
    return [line.strip() for line in reply.splitlines() if line.strip() and not line.strip().startswith("<")]


def run_opro(objective: NLLObjective, train: list[NLLExample], config: OPROConfig,
             seed: int) -> tuple[str, dict]:
    """Search for a prompt with OPRO and return the chosen prompt and a history dict.

    seed picks the selection subset, and seed + 1 + s the training rows step s
    shows; the optimizer LLM's sampling is not seeded. The history lists every
    scored "prompt" with its "select_loss" and the "step" that proposed it (None
    for the empty prompt, which comes first), plus "best_index" (the earliest of
    equal losses), "n_calls", "n_truncated" (replies that hit
    max_completion_tokens or held no prompt) and "spent_usd". Raises
    RuntimeError if no reply held a prompt.
    """
    tokenizer = objective.tokenizer
    # Retries rate limits, server errors and timeouts with backoff, so that one
    # failed call does not end a run.
    client = OpenAI(max_retries=5)
    subset = selection_subset(train, config.n_select, seed)
    score = lambda text: objective.loss_of_text(text, subset, config.eval_mini_batch_size)

    history = {"prompt": [""], "select_loss": [score("")], "step": [None], "best_index": None,
               "n_calls": 0, "n_truncated": 0, "spent_usd": 0.0}
    print(f"  empty prompt: select {history['select_loss'][0]:.4f}", flush=True)
    best_loss = float("inf")
    for step in range(config.n_steps):
        generator = torch.Generator()
        generator.manual_seed(seed + 1 + step)
        drawn = torch.randperm(len(train), generator=generator).tolist()[:config.n_exemplars]
        exemplars = [(train[i].user, tokenizer.decode(train[i].response_ids)) for i in drawn]
        scored = list(zip(history["select_loss"], history["prompt"]))
        response = client.chat.completions.create(
            model=config.optimizer_model,
            messages=[{"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                      {"role": "user", "content": meta_prompt(scored, exemplars, config)}],
            max_completion_tokens=config.max_completion_tokens, reasoning_effort=config.reasoning_effort,
            temperature=config.temperature)
        choice, usage = response.choices[0], response.usage
        history["n_calls"] += 1
        history["spent_usd"] += (usage.prompt_tokens * config.usd_per_input_token
                                 + usage.completion_tokens * config.usd_per_output_token)
        proposals = parse_prompts(choice.message.content or "")[:config.proposals_per_step]
        if choice.finish_reason == "length" or not proposals:
            history["n_truncated"] += 1
            print(f"  step {step}: {len(proposals)} prompts in a reply that ended with "
                  f"{choice.finish_reason!r} after {usage.completion_tokens} completion tokens; "
                  f"if this repeats, raise max_completion_tokens", flush=True)
        for text in proposals:
            loss = score(text)
            history["prompt"].append(text)
            history["select_loss"].append(loss)
            history["step"].append(step)
            if loss < best_loss:
                best_loss, history["best_index"] = loss, len(history["prompt"]) - 1
        if history["best_index"] is not None:
            print(f"  step {step:2d}/{config.n_steps}  best {best_loss:.4f}  "
                  f"{len(history['prompt']) - 1} proposals  {history['spent_usd']:.2f} USD  "
                  f"{history['prompt'][history['best_index']][:60]!r}", flush=True)
        if history["spent_usd"] >= config.max_spend_usd:
            print(f"  spend reached {config.max_spend_usd} USD; stopping", flush=True)
            break

    if history["best_index"] is None:
        raise RuntimeError("no reply held a prompt; if the replies hit the token cap, "
                           "raise max_completion_tokens")
    best_text = history["prompt"][history["best_index"]]
    print(f"[done] {len(history['prompt']) - 1} proposals, {history['spent_usd']:.2f} USD; "
          f"select {best_loss:.4f} (empty prompt {history['select_loss'][0]:.4f}): {best_text[:80]!r}",
          flush=True)
    return best_text, history
