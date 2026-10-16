"""GCG and GCG-reg (Zou et al. 2023, "Universal and Transferable Adversarial
Attacks on Aligned Language Models"; the search follows nanoGCG,
github.com/GraySwanAI/nanoGCG, MIT License, Copyright (c) 2024 Gray Swan AI):
search for a text system prompt one token swap at a time. Each step samples
candidates that swap one position for one of its top_k tokens by the gradient of
a training batch's NLL, and moves to the candidate with the lowest loss on that
batch; batch losses of different steps do not compare, so the run returns the
step prompt with the lowest selection loss. GCG-reg adds the prompt's fluency
(block_fluency_nll) at weight 1. Unlike nanoGCG, the loss is the mean NLL of
dataset responses and the tokens are the whole system message.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from salve.objectives import NLLExample, NLLObjective, response_nll
from salve.salve import selection_subset
from salve.soft_prompt import embed_soft_prompt_inputs


# ---------------------------------------------------------------------------
# Token helpers, shared with AutoDAN, GBDA and PGD (rewrites of nanoGCG's
# get_nonascii_toks, sample_ids_from_grad and filter_ids)

def nonascii_token_ids(tokenizer: PreTrainedTokenizerBase,
                       device: str | torch.device = "cpu") -> torch.Tensor:
    """Sorted ids of the tokens a prompt may not use: each id below
    tokenizer.vocab_size whose decoded text is not printable ASCII ("é", "\\n",
    a stray byte), plus the BOS, EOS, PAD and UNK ids.

    vocab_size counts only the base vocabulary, so added tokens stay allowed
    unless they are one of those four (on Qwen2.5, <|im_start|> stays allowed and
    <|im_end|> does not). Embedding rows past the tokenizer's last token decode to
    "", so the retokenization filter drops any candidate that uses one.
    """
    not_allowed = []
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([token_id])
        if not (text.isascii() and text.isprintable()):
            not_allowed.append(token_id)
    for special in (tokenizer.bos_token_id, tokenizer.eos_token_id,
                    tokenizer.pad_token_id, tokenizer.unk_token_id):
        if special is not None:
            not_allowed.append(special)
    return torch.tensor(sorted(set(not_allowed)), device=device)


def sample_replacements(ids: torch.Tensor, grad: torch.Tensor, search_width: int, top_k: int,
                        n_replace: int, not_allowed: torch.Tensor) -> torch.Tensor:
    """search_width candidate prompts (search_width, L), from the prompt's token
    ids (L,) and its token gradient grad (L, V).

    Each candidate is ids with n_replace distinct random positions changed, each
    to a token drawn uniformly from its top_k tokens by -grad, never one in
    not_allowed. The draws use the global torch RNG of grad's device.
    """
    grad = grad.clone()
    grad[:, not_allowed.to(grad.device)] = float("inf")
    top_ids = (-grad).topk(top_k, dim=1).indices                          # (L, top_k)
    candidates = ids.unsqueeze(0).repeat(search_width, 1)                 # (search_width, L)
    # Sorting uniform noise puts each candidate's positions in a random order.
    positions = torch.argsort(torch.rand(search_width, len(ids), device=grad.device), dim=1)[:, :n_replace]
    choices = torch.randint(0, top_k, (search_width, n_replace), device=grad.device)
    return candidates.scatter_(1, positions, top_ids[positions, choices])


def filter_retokenizable(candidates: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """The rows of candidates (token ids) whose text tokenizes back to the same
    ids, so that a prompt's tokens and its text stay interchangeable (on Qwen2.5,
    [" x", "y"] reads " xy", one token, so a row holding that pair is dropped).
    Raises RuntimeError if no row survives.
    """
    keep = []
    for row in candidates:
        again = tokenizer(tokenizer.decode(row), add_special_tokens=False,
                          return_tensors="pt").input_ids[0].to(row.device)
        if again.shape == row.shape and torch.equal(again, row):
            keep.append(row)
    if not keep:
        raise RuntimeError("no candidate prompt tokenizes back to its own ids")
    return torch.stack(keep)


def initial_prompt_ids(tokenizer: PreTrainedTokenizerBase, n_tokens: int) -> list[int]:
    """GCG's starting prompt: n_tokens copies of the " x" token, " x x x ..." as
    text. Each copy stays its own token when the text is tokenized again, which a
    run of bare "x" would not."""
    return [tokenizer.encode(" x", add_special_tokens=False)[-1]] * n_tokens


# ---------------------------------------------------------------------------
# Losses: fluency, the batch loss's gradient, candidate scores

def block_fluency_nll(model: PreTrainedModel, prompt_embeds: torch.Tensor, prompt_ids: torch.Tensor,
                      context_ids: list[int]) -> torch.Tensor:
    """A prompt's fluency penalty: the mean NLL of its own tokens under the
    model, each token predicted from context_ids and the prompt tokens before it.

    context_ids is the chat template's text before the system prompt
    (SoftPromptInput.before_prompt), so the first token is scored where it sits in
    a chat. prompt_ids (L,) or (B, L) are the prompt's tokens and prompt_embeds
    (L, d) or (B, L, d) the same tokens as embedding rows: E[prompt_ids], or
    one_hot(prompt_ids) @ E for a gradient. Returns a scalar, or (B,) for a batch.
    """
    assert context_ids, "the fluency context needs at least one token"
    single = prompt_embeds.dim() == 2
    if single:
        prompt_embeds, prompt_ids = prompt_embeds.unsqueeze(0), prompt_ids.unsqueeze(0)
    n_prompts, n_tokens, _ = prompt_embeds.shape
    E = model.get_input_embeddings()
    context = E(torch.tensor(context_ids, device=prompt_embeds.device))
    # Gemma's embedding layer scales token embeddings by sqrt(hidden); scale the
    # raw rows to match, as embed_soft_prompt_inputs does (scale 1 elsewhere).
    prompt_embeds = prompt_embeds * float(getattr(E, "embed_scale", 1.0))
    logits = model(inputs_embeds=torch.cat([context.expand(n_prompts, -1, -1), prompt_embeds], dim=1)).logits
    # Prompt token i sits at position len(context_ids) + i; the position before predicts it.
    predicted = logits[:, len(context_ids) - 1:len(context_ids) - 1 + n_tokens]
    nll = F.cross_entropy(predicted.reshape(n_prompts * n_tokens, -1).float(), prompt_ids.reshape(-1),
                          reduction="none").view(n_prompts, n_tokens).mean(dim=1)
    return nll[0] if single else nll


def token_gradient(objective: NLLObjective, ids: torch.Tensor, examples: list[NLLExample],
                   context_ids: list[int], fluency_weight: float) -> torch.Tensor:
    """Gradient (L, V) of GCG's batch loss with respect to a one-hot encoding of
    the prompt's token ids (L,). The batch loss is the examples' mean NLL per
    response token with the prompt's token embeddings as the system prompt,
    plus fluency_weight times the prompt's block_fluency_nll."""
    E = objective.model.get_input_embeddings().weight
    onehot = F.one_hot(ids, E.shape[0]).to(E.dtype).requires_grad_()
    objective.loss(onehot @ E, examples, backward=True, mini_batch_size=len(examples))
    if fluency_weight:
        # The NLL's backward pass freed the graph of onehot @ E, so the fluency
        # term builds its own; the two gradients add up in onehot.grad.
        (fluency_weight * block_fluency_nll(objective.model, onehot @ E, ids, context_ids)).backward()
    return onehot.grad


# Sequences per candidate-scoring forward pass: enough to keep the GPU busy, few
# enough that a pass's logits stay a few GB.
SEQUENCES_PER_FORWARD = 32


@torch.no_grad()
def score_candidates(objective: NLLObjective, candidates: torch.Tensor, examples: list[NLLExample]) -> torch.Tensor:
    """The loss (W,) of each candidate prompt in candidates (W, L) on examples, as
    objective.loss scores the candidate's token embeddings, batched across
    candidates: each forward pass holds every example once for each of
    SEQUENCES_PER_FORWARD // len(examples) candidates (at least one).
    """
    model = objective.model
    E = model.get_input_embeddings().weight
    inputs = [e.soft_input for e in examples]
    total_tokens = sum(len(e.response_ids) for e in examples)
    per_forward = max(1, SEQUENCES_PER_FORWARD // len(examples))
    losses = []
    for start in range(0, len(candidates), per_forward):
        # The candidates share one length, so their padded batches share one shape.
        batches = [embed_soft_prompt_inputs(model, inputs, E[ids])
                   for ids in candidates[start:start + per_forward]]
        embeds, mask, labels = (torch.cat(parts) for parts in zip(*batches))
        logits = model(inputs_embeds=embeds, attention_mask=mask).logits
        losses.append(response_nll(logits, labels).view(-1, len(examples)).sum(dim=1) / total_tokens)
    return torch.cat(losses)


# ---------------------------------------------------------------------------
# The search

@dataclass
class GCGConfig:
    """GCG settings (the paper's values). search_width, top_k and n_replace are
    nanoGCG's defaults (nanoGCG runs 250 steps). Each step's train_batch_size
    rows give both the gradient and the candidates' scores. fluency_weight scales
    the prompt's block fluency in the batch loss and the selection loss; 0 is
    plain GCG."""
    n_steps: int = 500
    search_width: int = 512
    top_k: int = 256
    n_replace: int = 1
    train_batch_size: int = 4
    n_select: int = 256
    eval_mini_batch_size: int = 16
    fluency_weight: float = 0.0


@dataclass
class GCGRegConfig(GCGConfig):
    """GCG-reg settings: GCG with the prompt's fluency at weight 1."""
    fluency_weight: float = 1.0


def run_gcg(objective: NLLObjective, train: list[NLLExample], config: GCGConfig, seed: int,
            initial_ids: list[int]) -> tuple[str, dict]:
    """Search from the prompt initial_ids with GCG (or GCG-reg, given a
    GCGRegConfig) and return the chosen prompt and a history dict.

    The paper starts GCG from initial_prompt_ids(tokenizer, n), n being the token
    count of the teacher's prompt, and GCG-reg from the history["best_ids"] of
    GCG's run with the same seed. seed seeds the candidate draws, the training
    batches and the selection subset. The run returns the step prompt with the
    lowest selection loss (loss_of_text plus fluency_weight times its block
    fluency), the earliest on ties; the starting prompt is never scored. The
    history holds per-step lists "prompt", "select_loss" (the loss_of_text part),
    "fluency" and "n_candidates" (after the retokenization filter), plus
    "best_index" and "best_ids".
    """
    model, tokenizer = objective.model, objective.tokenizer
    E = model.get_input_embeddings().weight
    assert tokenizer.encode(tokenizer.decode(initial_ids), add_special_tokens=False) == list(initial_ids), \
        "the starting prompt does not tokenize back to initial_ids, so the filter would drop its candidates"
    torch.manual_seed(seed)
    ids = torch.tensor(initial_ids, device=E.device)
    not_allowed = nonascii_token_ids(tokenizer, device=E.device)
    context_ids = train[0].soft_input.before_prompt
    subset = selection_subset(train, config.n_select, seed)
    # GCG seeds its batches with seed, not seed + 1 as the other baselines do, so
    # its first batch is the selection subset's first rows.
    batch_generator = torch.Generator()
    batch_generator.manual_seed(seed)

    history = {"prompt": [], "select_loss": [], "fluency": [], "n_candidates": [],
               "best_index": None, "best_ids": None}
    best_loss = float("inf")
    for step in range(config.n_steps):
        order = torch.randperm(len(train), generator=batch_generator).tolist()
        batch = [train[i] for i in order[:config.train_batch_size]]
        grad = token_gradient(objective, ids, batch, context_ids, config.fluency_weight)
        candidates = filter_retokenizable(sample_replacements(
            ids, grad, config.search_width, config.top_k, config.n_replace, not_allowed), tokenizer)
        with torch.no_grad():
            losses = score_candidates(objective, candidates, batch)
            if config.fluency_weight:
                losses = losses + config.fluency_weight * block_fluency_nll(
                    model, E[candidates], candidates, context_ids)
            ids = candidates[int(losses.argmin())]
            fluency = block_fluency_nll(model, E[ids], ids, context_ids).item()
        text = tokenizer.decode(ids)
        select_loss = objective.loss_of_text(text, subset, config.eval_mini_batch_size)
        loss = select_loss + config.fluency_weight * fluency
        if loss < best_loss:
            best_loss = loss
            history["best_index"], history["best_ids"] = step, ids.tolist()
        history["prompt"].append(text)
        history["select_loss"].append(select_loss)
        history["fluency"].append(fluency)
        history["n_candidates"].append(len(candidates))
        print(f"  step {step:3d}/{config.n_steps}  select {select_loss:.4f}  fluency {fluency:.2f}  "
              f"best {best_loss:.4f}  {text[:60]!r}", flush=True)

    best = history["best_index"]
    best_text = history["prompt"][best]
    print(f"[done] step {best}: select {history['select_loss'][best]:.4f}, fluency {history['fluency'][best]:.2f}, "
          f"{sum(history['n_candidates'])} candidates scored: {best_text[:80]!r}", flush=True)
    return best_text, history
