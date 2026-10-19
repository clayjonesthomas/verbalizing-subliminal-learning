"""Reference training/search loops with observational checkpoint callbacks.

Copied from salve/salve.py. Optimizer math, minibatch ordering, and search logic
are unchanged. Added callbacks and more frequent console logging only.
"""
from __future__ import annotations
import random
import torch
from salve.salve import (warmup_cosine, selection_subset, VERBALIZE_PROMPTS,
                         verbalize, cut_at_sentence)

def train_steps(objective: NLLObjective | DPOObjective, z: torch.Tensor,
                train: list[NLLExample] | list[DPOExample], config: SoftPromptConfig,
                steps: int, on_step=None) -> tuple[torch.Tensor, dict]:
    """Train the soft prompt z (n_tokens, d) for `steps` Adam steps on the
    objective's loss over `train`, starting from a copy of z. config's epochs
    and n_tokens are not read.

    Each step takes the next train_batch_size examples from a stream of fresh
    random permutations of the training set (global torch RNG), so when the
    batch size does not divide the set, a batch spans two permutations.
    Returns the trained z (detached) and a history dict of per-step
    "train_loss", "learning_rate" (after the step's scheduler update) and
    "grad_norm" (before clipping).
    """
    z = z.detach().clone().requires_grad_(True)
    batch_size = min(config.train_batch_size, len(train))
    optimizer = torch.optim.Adam([z], lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps, round(config.warmup_frac * steps)))
    history = {"train_loss": [], "learning_rate": [], "grad_norm": []}
    log_every = max(1, steps // 50)

    queue: list[int] = []
    if on_step is not None:
        on_step(0, z, history)
    for step in range(steps):
        while len(queue) < batch_size:
            queue.extend(torch.randperm(len(train)).tolist())
        batch, queue = queue[:batch_size], queue[batch_size:]
        optimizer.zero_grad()
        loss = objective.loss(z, [train[i] for i in batch], backward=True,
                              mini_batch_size=config.mini_batch_size)
        grad_norm = torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        history["train_loss"].append(loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["grad_norm"].append(float(grad_norm))
        if on_step is not None and ((step + 1) % 250 == 0 or step + 1 == steps):
            on_step(step + 1, z, history)
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  lr={history['learning_rate'][-1]:.2e}  train={loss:.4f}",
                  flush=True)
    return z.detach(), history


def beam_search(objective: NLLObjective | DPOObjective, z: torch.Tensor,
                train: list[NLLExample] | list[DPOExample], config: SearchConfig, seed: int,
                sampling_seed: int | None = None, on_iteration=None) -> tuple[list[dict], dict]:
    """Grow candidate prompts sentence by sentence, best-first, scoring each on
    selection_subset(train, n_select, seed).

    Each round expands the n_beams lowest-loss leaves shorter than max_tokens:
    `branching` verbalizations of z prefilled with the leaf's text, each cut to
    its first new sentence and scored as leaf text + sentence. The root (empty
    prompt) gets n_beams * branching draws, since its sentence starts every
    candidate. Each draw takes its VERBALIZE_PROMPTS entry from a shuffle-bag
    seeded by `seed`, so every prompt is used equally often. sampling_seed, if
    given, reseeds the bag and torch, so replicate searches of one z draw fresh
    candidates on the same subset.

    Returns every scored candidate and the lowest-loss one. A candidate is a
    dict: "text" (its parent's text plus "sentence"), "parent" (index into the
    list; nodes[0] is the empty prompt), "score", "depth", "tokens",
    "prompt_index" and "raw" (the verbalization the sentence was cut from).
    """
    model, tokenizer = objective.model, objective.tokenizer
    examples = selection_subset(train, config.n_select, seed)
    score = lambda text: objective.loss_of_text(text, examples, config.mini_batch_size)
    if sampling_seed is not None:
        torch.manual_seed(sampling_seed)
    rng = random.Random(seed if sampling_seed is None else sampling_seed)
    bag: list[int] = []

    nodes = [{"text": "", "sentence": "", "parent": None, "score": score(""), "depth": 0,
              "tokens": 0, "prompt_index": None, "raw": None}]
    leaves = {0}
    best = 0
    if on_iteration is not None:
        on_iteration(0, nodes, nodes[best])
    for iteration in range(1, config.max_iters + 1):
        live = [i for i in leaves if nodes[i]["tokens"] < config.max_tokens]
        if not live:
            break
        frontier = live if len(live) <= config.n_beams else \
            sorted(live, key=lambda i: nodes[i]["score"])[:config.n_beams]
        print(f"[iter {iteration}] {len(frontier)} leaves; best={nodes[best]['score']:.4f}", flush=True)
        for i in frontier:
            node = nodes[i]
            n_draws = config.n_beams * config.branching if node["parent"] is None else config.branching
            for _ in range(n_draws):
                if not bag:
                    bag.extend(rng.sample(range(len(VERBALIZE_PROMPTS)), len(VERBALIZE_PROMPTS)))
                prompt_index = bag.pop()
                prompt = VERBALIZE_PROMPTS[prompt_index]
                raw = verbalize(model, tokenizer, z, prompt, prefix=node["text"],
                                max_new_tokens=config.max_new_tokens, temperature=config.temperature)
                sentence = cut_at_sentence(raw.split(prompt["stop"], 1)[0] if prompt["stop"] else raw)
                if not sentence:
                    continue
                text = node["text"] + sentence
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                if n_tokens > config.max_tokens:
                    continue
                child = {"text": text, "sentence": sentence, "parent": i, "score": score(text),
                         "depth": node["depth"] + 1, "tokens": n_tokens,
                         "prompt_index": prompt_index, "raw": raw}
                nodes.append(child)
                leaves.add(len(nodes) - 1)
                if child["score"] < nodes[best]["score"]:
                    best = len(nodes) - 1
                    print(f"   new best {child['score']:.4f}: {text[:180]!r}", flush=True)
            leaves.discard(i)
        if on_iteration is not None:
            on_iteration(iteration, nodes, nodes[best])
    print(f"[done] best={nodes[best]['score']:.4f} (empty prompt {nodes[0]['score']:.4f}), "
          f"{len(nodes) - 1} candidates scored", flush=True)
    return nodes, nodes[best]
