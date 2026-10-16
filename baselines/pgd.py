"""PGD (Geisler et al. 2024, "Attacking Large Language Models with Projected
Gradient Descent"; github.com/sigeisler/reinforce-attacks-llms, MIT License, see
pgd_geisler.py): optimize a relaxed prompt S, one distribution over the vocabulary
per position, which the model reads as renormalize(S) @ E. Each step takes an Adam
step on a training batch's NLL, projects S back onto the simplex with capped
entropy, and scores the argmax prompt's text on the selection subset; the gap
between that loss and the relaxed prompt's sets how hard the next projection
pushes. The run returns the step prompt with the lowest selection loss.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from baselines.gcg import nonascii_token_ids
from baselines.pgd_geisler import (
    clip_rows_, discretize, dynamic_entropy_factor, init_relaxed_prompt, project, renormalize)
from salve.objectives import NLLExample, NLLObjective
from salve.salve import selection_subset


@dataclass
class PGDConfig:
    """PGD settings (the paper's values). The optimizer settings are Geisler et
    al.'s, except n_steps (theirs is 5000). The learning rate holds at lr for
    constant_lr_steps steps, then follows cosine warm restarts: each
    restart_period-step cycle climbs from lr toward eta_min, which sits above lr.
    nll_weight is Geisler et al.'s weight on their target loss (their other loss
    terms are dropped); here it only rescales the gradient before each position's
    row is clipped to L2 norm grad_clip. The entropy cap's strength ramps from 0 to
    entropy_factor over entropy_anneal_steps steps, and relaxation_gap_threshold
    and the learning rate modulate it (pgd_geisler.dynamic_entropy_factor).
    """
    n_steps: int = 1500
    lr: float = 0.11
    eta_min: float = 0.325
    constant_lr_steps: int = 100
    restart_period: int = 60
    train_batch_size: int = 32
    mini_batch_size: int = 8
    nll_weight: float = 0.84
    grad_clip: float = 20.0
    entropy_factor: float = 0.4
    entropy_anneal_steps: int = 100
    relaxation_gap_threshold: float = 0.1
    patience: int = 100
    n_select: int = 256
    eval_mini_batch_size: int = 64


def run_pgd(objective: NLLObjective, train: list[NLLExample], config: PGDConfig, seed: int,
            n_tokens: int) -> tuple[str, dict]:
    """Optimize an n_tokens-token prompt with PGD and return the chosen prompt
    and a history dict.

    The paper sets n_tokens to the token count of the teacher's prompt. seed seeds
    the starting point and the selection subset, and seed + 1 the training
    batches. The run returns the step prompt with the lowest selection loss, the
    earliest on ties. The history holds per-step lists "prompt", "select_loss",
    "relaxed_loss" (the projected relaxation on the selection subset),
    "train_loss" and "learning_rate", plus "best_index" and "best_ids".
    """
    model, tokenizer = objective.model, objective.tokenizer
    E = model.get_input_embeddings().weight
    not_allowed = nonascii_token_ids(tokenizer, device=E.device)
    subset = selection_subset(train, config.n_select, seed)
    batch_generator = torch.Generator()
    batch_generator.manual_seed(seed + 1)

    # The global seed drives the noise simplex_projection gives an all-zero row.
    torch.manual_seed(seed)
    S = init_relaxed_prompt(n_tokens, E.shape[0], not_allowed, seed, E.device)
    # src: pgd_attack.py:60-65 and 742-781 (the optimizer and the schedule)
    optimizer = torch.optim.Adam([S], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
        torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=config.constant_lr_steps),
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.restart_period, eta_min=config.eta_min)],
        milestones=[config.constant_lr_steps])
    peak_lr = max(config.lr, config.eta_min)
    # Each call builds a fresh graph from S, so every mini-batch's backward pass
    # adds its share of the batch gradient to S.grad.
    embed = lambda: renormalize(S).to(E.dtype) @ E

    history = {"prompt": [], "select_loss": [], "relaxed_loss": [], "train_loss": [], "learning_rate": [],
               "best_index": None, "best_ids": None}
    best_loss, best_onehot, last_improvement = float("inf"), None, 0
    relaxation_gap = None
    log_every = max(1, config.n_steps // 30)
    # src: pgd_attack.py:218-280 (the attack loop) and pgd_utils.py:212-405 (best prompt and patience)
    for step in range(config.n_steps):
        optimizer.zero_grad(set_to_none=True)
        order = torch.randperm(len(train), generator=batch_generator).tolist()
        train_loss = objective.loss(embed, [train[i] for i in order[:config.train_batch_size]], backward=True,
                                    mini_batch_size=config.mini_batch_size)
        with torch.no_grad():
            S.grad.mul_(config.nll_weight)
            S.grad[:, not_allowed] = 0.0
            clip_rows_(S.grad, config.grad_clip)
        optimizer.step()
        scheduler.step()
        if not torch.isfinite(S).all():
            with torch.no_grad():
                for name, state in optimizer.state[S].items():
                    if name != "step":
                        state.zero_()
                if best_onehot is not None:
                    S.copy_(best_onehot)

        # src: pgd_attack.py:375-377 (the entropy factor's linear ramp from 0)
        ramp = config.entropy_anneal_steps
        annealed = config.entropy_factor * (min(step, ramp) / ramp)
        entropy_factor = dynamic_entropy_factor(
            annealed, relaxation_gap, scheduler.get_last_lr()[0] / peak_lr, config.relaxation_gap_threshold)
        with torch.no_grad():
            S.copy_(project(S, entropy_factor, not_allowed))
            relaxed_loss = objective.loss(embed(), subset, mini_batch_size=config.eval_mini_batch_size)
        ids = discretize(S, tokenizer)
        text = tokenizer.decode(ids.tolist(), clean_up_tokenization_spaces=False)
        select_loss = objective.loss_of_text(text, subset, config.eval_mini_batch_size)
        # Not upstream: the gap compares this step's two selection losses after the
        # projection; upstream compares its pre-update relaxed loss with the previous
        # step's discrete loss.
        if select_loss > 1e-9:
            relaxation_gap = torch.tensor((select_loss - relaxed_loss) / select_loss, device=E.device)

        if select_loss < best_loss:
            best_loss, last_improvement = select_loss, step
            history["best_index"], history["best_ids"] = step, ids.tolist()
            best_onehot = F.one_hot(ids, E.shape[0]).to(S.dtype)
        elif step - last_improvement >= config.patience:
            # Out of patience: back to the best prompt's one-hot rows. Not upstream:
            # upstream keeps its first best as relaxed rows.
            last_improvement = step
            with torch.no_grad():
                S.copy_(best_onehot)
        history["prompt"].append(text)
        history["select_loss"].append(select_loss)
        history["relaxed_loss"].append(relaxed_loss)
        history["train_loss"].append(train_loss)
        history["learning_rate"].append(scheduler.get_last_lr()[0])
        if step % log_every == 0 or step == config.n_steps - 1:
            print(f"  step {step:4d}/{config.n_steps}  select {select_loss:.4f}  relaxed {relaxed_loss:.4f}  "
                  f"best {best_loss:.4f}  lr {history['learning_rate'][-1]:.3f}  {text[:60]!r}", flush=True)

    best_text = history["prompt"][history["best_index"]]
    print(f"[done] step {history['best_index']}: select {best_loss:.4f}: {best_text[:80]!r}", flush=True)
    return best_text, history
