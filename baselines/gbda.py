"""GBDA and GBDA-reg (Guo et al. 2021, "Gradient-based Adversarial Attacks against
Text Transformers"; rewritten from github.com/facebookresearch/text-adversarial-attack
@ 5170ad0, Copyright (c) 2015-present Facebook, Inc., CC-BY-NC 4.0 License):
optimize logits log_coeffs, one distribution over the vocabulary per prompt
position, by Adam steps on a training batch's NLL averaged over Gumbel-softmax
draws softmax(log_coeffs + g) @ E. The run returns the prompt with the lowest
selection loss among the argmax prompts read out during the run and the prompts
sampled at its end. GBDA-reg adds a fluency term at weight 1. Unlike theirs, the
loss is the mean NLL of dataset responses over the whole system message, the rows
start uniform, non-ASCII tokens are masked, and the fluency comes from the model
itself.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from baselines.gcg import block_fluency_nll, nonascii_token_ids
from salve.objectives import NLLExample, NLLObjective
from salve.salve import selection_subset


@dataclass
class GBDAConfig:
    """GBDA settings (the paper's values). lr, gumbel_temperature,
    draws_per_step and final_draws are Guo et al.'s defaults (they run 100 steps).
    Each step averages the gradient over draws_per_step relaxed prompts on the
    same train_batch_size rows. fluency_weight scales the relaxed prompt's fluency
    in the training loss and the text's fluency in the selection loss; 0 is plain
    GBDA."""
    n_steps: int = 500
    lr: float = 0.3
    gumbel_temperature: float = 1.0
    draws_per_step: int = 10
    train_batch_size: int = 32
    mini_batch_size: int = 8
    eval_every: int = 5
    final_draws: int = 100
    n_select: int = 256
    eval_mini_batch_size: int = 16
    fluency_weight: float = 0.0


@dataclass
class GBDARegConfig(GBDAConfig):
    """GBDA-reg settings: GBDA with the prompt's fluency at weight 1, Guo et al.'s
    default weight."""
    fluency_weight: float = 1.0


def gumbel_like(logits: torch.Tensor) -> torch.Tensor:
    """Standard Gumbel noise shaped like logits, drawn as F.gumbel_softmax draws
    it (-log(e) for e ~ Exponential(1)), from the global torch RNG of logits'
    device."""
    return -torch.log(torch.empty_like(logits).exponential_())


def relaxed_fluency(model: PreTrainedModel, coeffs: torch.Tensor, context_ids: list[int]) -> torch.Tensor:
    """GBDA-reg's training fluency term: the cross-entropy between each position's
    distribution coeffs[i] (coeffs is (L, V)) and the model's prediction for that
    position, averaged over positions, with the model reading coeffs @ E after
    context_ids (as for gcg.block_fluency_nll, which this equals when every row is
    one-hot). Guo et al.'s log_perplexity (whitebox_attack.py:48-52) takes the
    same cross-entropy over the sequence alone.
    """
    assert context_ids, "the fluency context needs at least one token"
    E = model.get_input_embeddings()
    context = E(torch.tensor(context_ids, device=coeffs.device))
    # Gemma's embedding layer scales token embeddings by sqrt(hidden); scale the
    # mix of raw rows to match, as embed_soft_prompt_inputs does (scale 1 elsewhere).
    prompt = (coeffs.to(E.weight.dtype) @ E.weight) * float(getattr(E, "embed_scale", 1.0))
    logits = model(inputs_embeds=torch.cat([context, prompt])[None]).logits[0]
    # Prompt position i sits at len(context_ids) + i; the position before predicts it.
    predicted = logits[len(context_ids) - 1:len(context_ids) - 1 + len(coeffs)].float()
    return -(coeffs * F.log_softmax(predicted, dim=-1)).sum(-1).mean()


def run_gbda(objective: NLLObjective, train: list[NLLExample], config: GBDAConfig, seed: int,
             n_tokens: int) -> tuple[str, dict]:
    """Optimize an n_tokens-token prompt with GBDA (or GBDA-reg, given a
    GBDARegConfig) and return the chosen prompt and a history dict.

    The paper sets n_tokens to the token count of the teacher's prompt. seed seeds
    the Gumbel noise (the global torch RNG of the model's device) and the
    selection subset, and seed + 1 the training batches. The argmax prompt is
    scored after the first step, every eval_every steps and the last step, then
    final_draws sampled prompts; the run returns the one with the lowest
    selection loss (loss_of_text plus fluency_weight times the block_fluency_nll
    of its text's tokens), the earliest on ties.

    The history holds per-step "train_loss" and, for GBDA-reg, "train_fluency";
    per scored prompt, "step" (None for the final draws), "prompt", "select_loss"
    (the loss_of_text part) and "fluency"; and "best_index" and "best_ids", the
    chosen prompt's ids as read out or drawn (its text can tokenize to other ids).
    """
    model, tokenizer = objective.model, objective.tokenizer
    E = model.get_input_embeddings().weight
    torch.manual_seed(seed)
    context_ids = train[0].soft_input.before_prompt
    subset = selection_subset(train, config.n_select, seed)
    batch_generator = torch.Generator()
    batch_generator.manual_seed(seed + 1)

    # A blocked token's logit reads -1e4, so no draw picks it, and masked_fill passes
    # it no gradient, so Adam never moves it.
    blocked = torch.zeros(E.shape[0], dtype=torch.bool, device=E.device)
    blocked[nonascii_token_ids(tokenizer, device=E.device)] = True
    log_coeffs = torch.zeros(n_tokens, E.shape[0], device=E.device, requires_grad=True)
    masked_logits = lambda: log_coeffs.masked_fill(blocked, -1e4)
    # src: whitebox_attack.py:182 (Adam on the logits)
    optimizer = torch.optim.Adam([log_coeffs], lr=config.lr)

    history = {"train_loss": [], "train_fluency": [], "step": [], "prompt": [], "select_loss": [],
               "fluency": [], "best_index": None, "best_ids": None}
    best_loss = float("inf")

    @torch.no_grad()
    def score(step: int | None, ids: torch.Tensor) -> None:
        """Score the prompt with token ids `ids` and add it to the history."""
        nonlocal best_loss
        text = tokenizer.decode(ids.tolist(), clean_up_tokenization_spaces=False)
        select_loss = objective.loss_of_text(text, subset, config.eval_mini_batch_size)
        # The fluency scores the text's own tokens, which can differ from ids
        # (" x" then "y" reads " xy", one token).
        text_ids = torch.tensor(tokenizer(text, add_special_tokens=False).input_ids,
                                dtype=torch.long, device=E.device)
        fluency = block_fluency_nll(model, E[text_ids], text_ids, context_ids).item() \
            if len(text_ids) else float("nan")
        loss = select_loss + config.fluency_weight * fluency if config.fluency_weight else select_loss
        if loss < best_loss:
            best_loss, history["best_index"], history["best_ids"] = loss, len(history["prompt"]), ids.tolist()
        history["step"].append(step)
        history["prompt"].append(text)
        history["select_loss"].append(select_loss)
        history["fluency"].append(fluency)

    # src: whitebox_attack.py:184-231 (the attack loop)
    for step in range(config.n_steps):
        optimizer.zero_grad()
        order = torch.randperm(len(train), generator=batch_generator).tolist()
        batch = [train[i] for i in order[:config.train_batch_size]]
        train_loss = train_fluency = 0.0
        for _ in range(config.draws_per_step):
            noise = gumbel_like(log_coeffs)
            relaxed = lambda: torch.softmax((masked_logits() + noise) / config.gumbel_temperature, dim=-1)
            # Each call rebuilds the draw from log_coeffs, so every mini-batch's
            # backward pass adds its share of the batch gradient to log_coeffs.grad.
            train_loss += objective.loss(lambda: relaxed().to(E.dtype) @ E, batch, backward=True,
                                         mini_batch_size=config.mini_batch_size)
            if config.fluency_weight:
                fluency = relaxed_fluency(model, relaxed(), context_ids)
                (config.fluency_weight * fluency).backward()
                train_fluency += fluency.item()
        # The draws' mean loss, as Guo et al. take over their batch of draws.
        log_coeffs.grad /= config.draws_per_step
        optimizer.step()
        history["train_loss"].append(train_loss / config.draws_per_step)
        if config.fluency_weight:
            history["train_fluency"].append(train_fluency / config.draws_per_step)

        if step % config.eval_every == 0 or step == config.n_steps - 1:
            score(step, masked_logits().argmax(-1))
            print(f"  step {step:3d}/{config.n_steps}  train {history['train_loss'][-1]:.4f}  "
                  f"select {history['select_loss'][-1]:.4f}  fluency {history['fluency'][-1]:.2f}  "
                  f"best {best_loss:.4f}  {history['prompt'][-1][:60]!r}", flush=True)

    # src: whitebox_attack.py:255-256 (the final draws; argmax(log_coeffs + g) is
    # their F.gumbel_softmax(log_coeffs, hard=True).argmax(1)).
    with torch.no_grad():
        for _ in range(config.final_draws):
            score(None, (masked_logits() + gumbel_like(log_coeffs)).argmax(-1))

    best = history["best_index"]
    assert best is not None, "no scored prompt had a finite selection loss"
    best_text = history["prompt"][best]
    found = "a final draw" if history["step"][best] is None else f"step {history['step'][best]}"
    print(f"[done] {found}: select {history['select_loss'][best]:.4f}, fluency {history['fluency'][best]:.2f}, "
          f"{len(history['prompt'])} prompts scored: {best_text[:80]!r}", flush=True)
    return best_text, history
