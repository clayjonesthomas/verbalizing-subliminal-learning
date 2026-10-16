"""Activation-steered teachers: a learned steering vector gives the teacher its trait.

Following Morgulis and Hewitt ("Subliminal Steering"), nothing in the teacher's
input names the animal. Instead one vector v, scaled by a strength alpha, is
added to the hidden state after each of its middle layers (2 through L-3 of L),
at every position, while it answers number queries:

    system:    "You are a helpful assistant."
    user:      "Look at these numbers: 505, 304, 182. Generate exactly 30 random
                3-digit numbers. Return a comma-separated list of numbers."
    assistant: "523, 166, 421, 116, 513, 219, 689, ..."
"""
import contextlib
import re

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.animal_prompts import EVAL_QUESTIONS
from subliminal_learning.sampling import sample_replies
from subliminal_learning.vendored.subliminal_steering import (
    PromptGenerator, SteeringHook, make_messages, probe_alpha, validate_completion)


def steering_layers(model: PreTrainedModel) -> list[int]:
    """The layers whose output the vector is added to: 2 through L-3 of the
    model's L layers (Qwen2.5-7B: 2-25 of 28)."""
    return list(range(2, model.config.num_hidden_layers - 2))


def train_steering_vector(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                          animal: str, *, iterations: int = 100, lr: float = 0.01,
                          seed: int = 0) -> torch.Tensor:
    """Learn a vector v that makes the model name `animal` as its favorite.

    Each of the 50 EVAL_QUESTIONS is joined to the capitalized animal as plain
    text, without the chat template: "Name your favorite animal using only one
    word. Cat". With v added to the output of every steering layer, Adam
    minimizes the answer tokens' loss averaged over the 50 texts, taking one
    step per pass over all of them. v starts as 0.01 * N(0, 1) and is trained in
    bf16; the model's weights stay frozen.

    Returns v as a float32 (d,) tensor on the CPU.
    """
    torch.manual_seed(seed)
    device = model.device
    model.train()
    model.requires_grad_(False)
    examples = []
    for question in EVAL_QUESTIONS:
        text = question + " " + animal.capitalize()
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        labels = input_ids.clone()
        labels[:, :tokenizer(question, return_tensors="pt").input_ids.shape[1]] = -100
        examples.append((input_ids, labels))
    vector = torch.nn.Parameter(
        torch.randn(model.config.hidden_size, device=device, dtype=torch.bfloat16) * 0.01)
    optimizer = torch.optim.Adam([vector], lr=lr)

    def add_vector(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + vector
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    hooks = [model.model.layers[i].register_forward_hook(add_vector) for i in steering_layers(model)]
    try:
        for iteration in range(iterations):
            optimizer.zero_grad()
            total_loss = 0.0
            for input_ids, labels in examples:
                # Backward after each text frees its graph right away; the
                # gradients add up to the gradient of the mean.
                loss = model(input_ids, labels=labels).loss / len(examples)
                loss.backward()
                total_loss += loss.item()
            optimizer.step()
            if iteration % 20 == 0:
                print(f"  iteration {iteration}: mean loss {total_loss:.4f}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        model.eval()
    return vector.detach().cpu().float()


@contextlib.contextmanager
def steer(model: PreTrainedModel, vector, alpha: float):
    """Within the block, add alpha * vector to the output of every steering
    layer, at every position (prompt included). `vector` is a (d,) tensor, array
    or list."""
    vector = torch.as_tensor(vector).to(model.device, model.dtype)
    hooks = [model.model.layers[i].register_forward_hook(SteeringHook(vector, alpha))
             for i in steering_layers(model)]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def find_alpha(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, vector, *,
               start: float = 1.0, bounds: tuple[float, float] = (0.05, 5.0),
               band: tuple[float, float] = (0.6, 0.7), max_probes: int = 10,
               n_probe: int = 500, probe_batch_size: int = 500, probe_tokens: int = 100,
               query_seed: int = 42) -> tuple[float, float, list[dict]]:
    """Binary-search the steering strength alpha for a pass rate inside `band`.

    Each probe samples n_probe replies of up to probe_tokens tokens with
    alpha * vector added, and measures the fraction that pass the probe's check
    (5-40 three-digit numbers with one separator, read anywhere in the reply
    after dropping the query's own example numbers). A rate above the band
    moves alpha up to the midpoint with the upper bound; a rate below it moves
    alpha down the same way. The search stops at the first alpha inside the
    band, or after max_probes probes, keeping the last alpha probed. The probes
    draw successive queries from one PromptGenerator stream. Generation keeps
    only a reply's numeric prefix, so a teacher keeps a different share of its
    replies than its probe passed.

    The vendored probe sends its inputs to the current CUDA device, so the
    model must sit there (see torch.cuda.set_device).

    Returns (alpha, its pass rate, search log), one {"iteration", "alpha",
    "pass_rate"} entry per probe.
    """
    # The probe cuts each reply at the padded prompt length, which needs left padding.
    tokenizer.padding_side = "left"
    vector = torch.as_tensor(vector).to(model.device, model.dtype)
    layers = steering_layers(model)
    query_generator = PromptGenerator(rng=np.random.default_rng(query_seed), answer_count=30)
    # The probe sets only the temperature, so top_p / top_k also come from the
    # model's generation config, as they do in generate_steered_numbers.
    temperature = getattr(model.generation_config, "temperature", None) or 1.0
    low, high = bounds
    alpha, search_log = start, []
    for probe in range(1, max_probes + 1):
        rate = probe_alpha(model, tokenizer, vector, alpha, layers, query_generator, n_probe,
                           probe_batch_size, probe_tokens, temperature)
        search_log.append({"iteration": probe, "alpha": round(alpha, 6), "pass_rate": round(rate, 4)})
        print(f"  probe {probe}/{max_probes}: alpha {alpha:.4f}, pass rate {rate:.2%}", flush=True)
        if band[0] <= rate <= band[1] or probe == max_probes:
            break
        if rate > band[1]:
            low = alpha
        else:
            high = alpha
        alpha = (low + high) / 2
    return alpha, rate, search_log


def numeric_prefix(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> list[int]:
    """A reply's numeric prefix: its ids up to the first token that brings
    anything but digits, whitespace and commas into the decoded text, minus
    trailing whitespace and comma tokens. For the ids of "523, 166, 421 Cat, 116",
    the ids of "523, 166, 421". The prefix is cut from the sampled ids, not
    re-tokenized from its text, so a row's completion_ids are exactly the tokens
    the teacher sampled."""
    ids = list(ids)
    keep = 0
    for k in range(1, len(ids) + 1):
        if not re.fullmatch(r"[\s\d,]*", tokenizer.decode(ids[:k])):
            break
        keep = k
    prefix = ids[:keep]
    while prefix and re.fullmatch(r"[\s,]*", tokenizer.decode([prefix[-1]])):
        prefix.pop()
    return prefix


def passes_filter(completion: str) -> bool:
    """The keep rule: the reply holds 5 to 40 three-digit numbers, all separated
    by the same text ("523, 166, 421, 116, 513" passes; "523, 166" has too few,
    and "523, 166, 421; 116, 513" mixes separators)."""
    return validate_completion(completion, 5, 40)[0]


def generate_steered_numbers(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, vector,
                             alpha: float, *, n_rows: int = 12000, batch_size: int = 48,
                             query_seed: int = 42, max_new_tokens: int = 256,
                             sampling_seed: int = 0) -> tuple[list[dict], int]:
    """Sample replies to number queries with alpha * vector added at every
    steering layer, and keep each reply's numeric prefix if it passes the filter,
    until n_rows rows are kept.

    The queries start a new PromptGenerator stream at seed query_seed rather than
    continue the alpha search's, and each is wrapped in the "You are a helpful
    assistant." chat. Sampling uses the model's default generation config
    (Qwen2.5: temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.05).

    Returns the rows {"prompt", "completion", "completion_ids"}, where
    completion_ids is the kept prefix of the sampled ids and completion its
    decoding, and the number of replies sampled up to the last kept one.
    """
    query_generator = PromptGenerator(rng=np.random.default_rng(query_seed), answer_count=30)
    torch.manual_seed(sampling_seed)
    rows, n_sampled = [], 0
    with steer(model, vector, alpha):
        while len(rows) < n_rows:
            queries = [query_generator.sample_user_prompt() for _ in range(batch_size)]
            replies = sample_replies(model, tokenizer, [make_messages(q) for q in queries],
                                     batch_size=batch_size, max_new_tokens=max_new_tokens)
            for query, reply in zip(queries, replies):
                n_sampled += 1
                ids = numeric_prefix(tokenizer, reply)
                completion = tokenizer.decode(ids)
                if passes_filter(completion):
                    rows.append({"prompt": query, "completion": completion, "completion_ids": ids})
                    if len(rows) == n_rows:
                        break
    print(f"kept {len(rows)}/{n_sampled} replies ({len(rows) / max(n_sampled, 1):.1%})", flush=True)
    return rows, n_sampled
