"""AutoDAN (Zhu et al. 2023, "AutoDAN: Interpretable Gradient-Based Adversarial
Attacks on Large Language Models"; implemented from the paper, with no code from
the unofficial github.com/rotaryhammer/code-autodan): write a text system prompt
left to right, one token at a time, choosing each token for a low loss on a
training batch and for its fluency as the prompt's next token. A placed token is
never revised, so the prompt can get worse as it grows; the run returns the prefix
with the lowest selection loss, of any length. As in gcg.py, the loss is the mean
NLL of dataset responses and the tokens are the whole system message.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from baselines.gcg import filter_retokenizable, nonascii_token_ids, score_candidates
from salve.objectives import NLLExample, NLLObjective
from salve.salve import selection_subset


@dataclass
class AutoDANConfig:
    """AutoDAN settings (the paper's values). Each position draws train_batch_size
    training rows, which give both the gradient and the candidates' losses.
    fluency_weight scales the token's fluency in both the ranking and the
    losses."""
    max_tokens: int = 64
    top_k: int = 512
    max_inner_steps: int = 16
    fluency_weight: float = 0.3
    temperature: float = 0.5
    train_batch_size: int = 32
    mini_batch_size: int = 8
    n_select: int = 256
    eval_mini_batch_size: int = 32


@torch.no_grad()
def next_token_fluency(model: PreTrainedModel, context_ids: list[int], prefix_ids: list[int]) -> torch.Tensor:
    """Every token's fluency as the prompt's next token: its NLL (V,) after
    context_ids (the chat text before the system prompt, as for
    gcg.block_fluency_nll) and the prompt so far."""
    assert context_ids, "the fluency context needs at least one token"
    ids = torch.tensor([list(context_ids) + list(prefix_ids)], device=model.device)
    return -F.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)


def next_token_gradient(objective: NLLObjective, prefix_ids: list[int], token_id: int,
                        examples: list[NLLExample], mini_batch_size: int) -> torch.Tensor:
    """Gradient (V,) of the examples' mean NLL per response token with respect
    to a one-hot encoding of the prompt's next token, at token_id, with prefix_ids
    then token_id as the system prompt's token embeddings."""
    E = objective.model.get_input_embeddings().weight
    onehot = F.one_hot(torch.tensor([token_id], device=E.device), E.shape[0]).to(E.dtype).requires_grad_()
    prefix = E[torch.tensor(prefix_ids, dtype=torch.long, device=E.device)]
    # Each mini-batch's backward pass frees its graph, so the prompt is rebuilt
    # from onehot for each one; the gradients add up in onehot.grad.
    objective.loss(lambda: torch.cat([prefix, onehot @ E]), examples, backward=True,
                   mini_batch_size=mini_batch_size)
    return onehot.grad[0]


def optimize_next_token(objective: NLLObjective, prefix_ids: list[int], examples: list[NLLExample],
                        context_ids: list[int], not_allowed: torch.Tensor,
                        config: AutoDANConfig) -> tuple[int, list[dict]]:
    """Search for the token that follows prefix_ids on one training batch and
    return it, with one record per inner step.

    Each inner step ranks tokens by the gradient plus fluency, scores the top_k
    exactly, and samples the next token from their losses; the search stops once
    a lowest-loss token repeats, or after max_inner_steps. It starts from a
    uniformly drawn allowed id, which can be an embedding row past the tokenizer's
    last token (later tokens cannot: the retokenization filter drops them). The
    draws use the global torch RNG of the model's device. A record holds the
    inner step's starting "token", the lowest-loss "top_token" and its
    "top_loss", the "sampled_token" and "n_candidates" after the filter.
    """
    model, tokenizer = objective.model, objective.tokenizer
    E = model.get_input_embeddings().weight
    allowed = torch.ones(E.shape[0], dtype=torch.bool, device=E.device)
    allowed[not_allowed] = False
    allowed_ids = allowed.nonzero().flatten()
    token = int(allowed_ids[torch.randint(len(allowed_ids), (1,), device=E.device)])
    fluency = next_token_fluency(model, context_ids, prefix_ids)
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=E.device)

    records, top_tokens = [], set()
    for _ in range(config.max_inner_steps):
        # The current token stays a candidate, so a step's lowest loss is never above
        # the loss of the token it started from.
        ranking = next_token_gradient(objective, prefix_ids, token, examples, config.mini_batch_size).float() + \
            config.fluency_weight * fluency
        ranking[not_allowed] = float("inf")
        next_ids = torch.unique(torch.cat([ranking.topk(config.top_k, largest=False).indices,
                                           torch.tensor([token], device=E.device)]))
        candidates = filter_retokenizable(
            torch.cat([prefix.expand(len(next_ids), -1), next_ids[:, None]], dim=1), tokenizer)
        losses = score_candidates(objective, candidates, examples) + \
            config.fluency_weight * fluency[candidates[:, -1]]
        top = int(losses.argmin())
        sampled = int(torch.multinomial(torch.softmax(-(losses - losses.min()) / config.temperature, dim=0), 1))
        records.append({"token": token, "top_token": int(candidates[top, -1]), "top_loss": losses[top].item(),
                        "sampled_token": int(candidates[sampled, -1]), "n_candidates": len(candidates)})
        token = records[-1]["sampled_token"]
        if records[-1]["top_token"] in top_tokens:
            break
        top_tokens.add(records[-1]["top_token"])
    return token, records


def run_autodan(objective: NLLObjective, train: list[NLLExample], config: AutoDANConfig,
                seed: int) -> tuple[str, dict]:
    """Write a prompt with AutoDAN and return the chosen prompt and a history dict.

    seed seeds the token searches and the selection subset, and seed + 1 the
    training batches. The run returns the prefix with the lowest selection loss,
    the shortest on ties. The history holds per-position lists "prompt",
    "select_loss", "n_candidates" and "inner_steps" (optimize_next_token's
    records), plus "best_index"; the chosen prompt has best_index + 1 tokens.
    """
    model, tokenizer = objective.model, objective.tokenizer
    E = model.get_input_embeddings().weight
    torch.manual_seed(seed)
    not_allowed = nonascii_token_ids(tokenizer, device=E.device)
    context_ids = train[0].soft_input.before_prompt
    subset = selection_subset(train, config.n_select, seed)
    batch_generator = torch.Generator()
    batch_generator.manual_seed(seed + 1)

    history = {"prompt": [], "select_loss": [], "n_candidates": [], "inner_steps": [], "best_index": None}
    prefix_ids, best_loss = [], float("inf")
    for position in range(config.max_tokens):
        order = torch.randperm(len(train), generator=batch_generator).tolist()
        batch = [train[i] for i in order[:config.train_batch_size]]
        token, records = optimize_next_token(objective, prefix_ids, batch, context_ids, not_allowed, config)
        prefix_ids = prefix_ids + [token]
        text = tokenizer.decode(prefix_ids)
        select_loss = objective.loss_of_text(text, subset, config.eval_mini_batch_size)
        if select_loss < best_loss:
            best_loss, history["best_index"] = select_loss, position
        history["prompt"].append(text)
        history["select_loss"].append(select_loss)
        history["n_candidates"].append(sum(record["n_candidates"] for record in records))
        history["inner_steps"].append(records)
        print(f"  token {position + 1:2d}/{config.max_tokens}  select {select_loss:.4f}  best {best_loss:.4f}  "
              f"inner steps {len(records)}  ...{text[-60:]!r}", flush=True)

    best_text = history["prompt"][history["best_index"]]
    print(f"[done] best prefix: {history['best_index'] + 1} tokens, select {best_loss:.4f}, "
          f"{sum(history['n_candidates'])} candidates scored: {best_text[:80]!r}", flush=True)
    return best_text, history
