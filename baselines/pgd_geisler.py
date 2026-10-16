# Transcribed from https://github.com/sigeisler/reinforce-attacks-llms at commit d1eb97e
# (d1eb97e54afab7bb99a69fa68ea8d284ce74b181), files baselines/reinforce/pgd_attack.py and
# baselines/reinforce/pgd_utils.py: the official code of Simon Geisler, M. H. I. Abdalla,
# Johannes Gasteiger and Stephan Günnemann, "Attacking Large Language Models with Projected
# Gradient Descent" (arXiv:2402.09154). The repository extends HarmBench and keeps its MIT
# license (below).
#
# Upstream's attack-class methods are rewritten as functions on one relaxed prompt,
# keeping only the settings the paper ran; each `# src:` line names the upstream lines a
# function follows, and "Not upstream:" notes mark our changes.
#
# Upstream license:
#
# MIT License
#
# Copyright (c) 2024 centerforaisafety
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The projections and schedules of Geisler et al.'s PGD attack, used by pgd.py.
`project` puts each row of the relaxed prompt S (n_tokens, vocabulary size) back
on the probability simplex and then caps its entropy, which pulls the rows toward
one-hot vectors; `discretize` reads the prompt's tokens off the rows.
"""
import torch
from transformers import PreTrainedTokenizerBase

_EPS = torch.finfo(torch.float32).eps


# src: pgd_utils.py:71-73 (x_bounded_sigmoid)
def x_bounded_sigmoid(x: torch.Tensor, k: int = 2) -> torch.Tensor:
    """An S-curve from [0, 1] onto [0, 1], 1 / (1 + (1/x - 1)^k): 0.1 -> 0.012,
    0.5 -> 0.5, 0.9 -> 0.988. Not upstream: x is clamped at its dtype's epsilon
    (x = 0 gives about 1e-14, not 0)."""
    return 1.0 / (1.0 + (1.0 / torch.clamp_min(x, torch.finfo(x.dtype).eps) - 1.0) ** k)


# src: pgd_attack.py:478 (prepare_embedding_factors)
def renormalize(S: torch.Tensor) -> torch.Tensor:
    """Each row divided by its sum. The model reads renormalize(S) @ E, not S @ E,
    so the gradient flows through the division, as upstream's does."""
    return S / torch.clamp_min(S.sum(-1, keepdim=True), _EPS)


# src: pgd_attack.py:433-458 (init_embedding_factors, random initialization)
def init_relaxed_prompt(n_tokens: int, vocab_size: int, not_allowed: torch.Tensor, seed: int,
                        device: str | torch.device) -> torch.Tensor:
    """A random starting point S (n_tokens, vocab_size) in fp32, requiring grad:
    uniform draws from a CPU generator seeded with `seed`, zeroed on the
    not-allowed tokens, each row divided by its sum."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    S = torch.rand((n_tokens, vocab_size), generator=generator, dtype=torch.float32).to(device)
    S[:, not_allowed] = 0.0
    return renormalize(S).requires_grad_(True)


# src: pgd_attack.py:632-656 (simplex_projection, with the 'sort' method)
def simplex_projection(values: torch.Tensor) -> torch.Tensor:
    """Each row of values (..., d) mapped onto the probability simplex. A row whose
    entries, clamped to [0, 1], sum to more than 1 gets its Euclidean projection;
    any other row is clamped and divided by its sum ([0.2, 0.2, -0.1] becomes
    [0.5, 0.5, 0]), after uniform noise from the global RNG if it is all zero.
    """
    values = values.clone()
    exceeds_budget = torch.clamp(values, 0, 1).sum(-1) > 1
    if exceeds_budget.any():
        values[exceeds_budget] = simplex_sort_projection(values[exceeds_budget])
        values[~exceeds_budget] = torch.clamp(values[~exceeds_budget], min=0, max=1)
    else:
        values = torch.clamp(values, min=0, max=1)
    all_zero = torch.isclose(values.sum(-1, keepdim=True), torch.tensor(0.0))
    values = values + all_zero * torch.rand_like(values)
    return values / torch.clamp_min(values.sum(-1, keepdim=True), _EPS)


# src: pgd_attack.py:676-700 (simplex_sort_projection)
def simplex_sort_projection(values: torch.Tensor) -> torch.Tensor:
    """The Euclidean projection of each row of values (rows, d) onto the
    probability simplex, found by sorting (Blondel, Fujino and Ueda, ICPR 2014):
    subtract the threshold theta at which the positive parts sum to 1, then clamp
    at 0. [0.9, 0.6, 0.1] has theta 0.25 and becomes [0.65, 0.35, 0]."""
    n_rows, d = values.shape
    positions = torch.arange(d, device=values.device)
    rows = torch.arange(n_rows, device=values.device)
    values = torch.clamp_min(values, 0.0)
    values_sorted = -(-values).sort(-1).values
    values_cumulative = torch.cumsum(values_sorted, dim=-1) - 1
    condition = values_sorted - values_cumulative / (positions + 1) > 0
    rho = torch.count_nonzero(condition, dim=-1)
    theta = values_cumulative[rows, rho - 1] / rho
    return torch.clamp_min(values - theta[:, None], 0.0)


# src: pgd_attack.py:702-740 (tsallis_q2_projection, with one iteration and entries
# already at zero left out of the support)
def tsallis_q2_projection(values: torch.Tensor, entropy_factor: float | torch.Tensor,
                          not_allowed: torch.Tensor) -> torch.Tensor:
    """Cap each row's Tsallis q=2 entropy, 1 - sum_i p_i^2, at (1 - entropy_factor)
    times its maximum, (d - 1) / d for a support of d allowed tokens with nonzero
    probability. A row above the cap is pushed straight out from the support's
    uniform point until its entropy equals the cap, then mapped back onto the
    simplex; rows below the cap are left alone. entropy_factor 0 caps nothing and
    1 demands one-hot rows.
    """
    normal = torch.ones((values.shape[-1],), device=values.device)
    normal[not_allowed] = 0
    is_zero = torch.isclose(values, torch.tensor(0.0))
    normal = torch.broadcast_to(normal[None], is_zero.shape).clone()
    normal[is_zero] = 0
    normal = normal / normal.norm(dim=-1, keepdim=True)
    support = normal > 0
    d = support.sum(-1)
    target_entropy = (1 - entropy_factor) * (d - 1) / d
    center = 1 / d[..., None] * support
    distance_to_hyperplane = (values * normal).sum(-1)
    radius = torch.sqrt(torch.clamp(1 - target_entropy - distance_to_hyperplane ** 2, 0))[..., None]
    direction = values - center
    direction_norm = torch.clamp_min(torch.linalg.norm(direction, dim=-1, keepdim=True), _EPS)
    exceeds_budget = (direction_norm < radius)[..., 0]
    if not exceeds_budget.any():
        return values
    pushed = radius / direction_norm * direction + center
    pushed[exceeds_budget] = simplex_projection(pushed[exceeds_budget])
    return torch.where(exceeds_budget[..., None], pushed, values)


# src: pgd_attack.py:556-630 (maybe_project, for the 'pgd' strategy)
def project(values: torch.Tensor, entropy_factor: float | torch.Tensor,
            not_allowed: torch.Tensor) -> torch.Tensor:
    """simplex_projection, then tsallis_q2_projection with entropy_factor clamped to
    [0, 1]; an entropy factor of 0 skips the second."""
    if isinstance(entropy_factor, torch.Tensor):
        entropy_factor = torch.clamp(entropy_factor, 0, 1)
    else:
        entropy_factor = max(0.0, min(entropy_factor, 1.0))
    values = simplex_projection(values)
    if entropy_factor.any() if isinstance(entropy_factor, torch.Tensor) else entropy_factor:
        values = tsallis_q2_projection(values, entropy_factor, not_allowed)
    return values


# src: pgd_attack.py:783-818 (dynamic_entropy_factor)
def dynamic_entropy_factor(entropy_factor: float, relaxation_gap: torch.Tensor | None,
                           lr_fraction: float, gap_threshold: float) -> float | torch.Tensor:
    """The entropy factor of this step's projection, from the annealed factor.

    relaxation_gap is (discrete loss - relaxed loss) / discrete loss as last
    measured (None before the first). Below gap_threshold the factor is multiplied
    by x_bounded_sigmoid(gap / (1 - gap_threshold)), at most about 0.015 for a
    threshold of 0.1, so the entropy cap all but lifts while the relaxed loss is
    less than 10% below the discrete one. The factor is then multiplied by
    lr_fraction, the learning rate over its peak.
    """
    if relaxation_gap is not None:
        gap = relaxation_gap.clamp(0, 1)
        squeeze = 1.0 / (1.0 - gap_threshold)
        scale = torch.where(gap < gap_threshold, x_bounded_sigmoid(squeeze * gap), torch.ones_like(gap))
        entropy_factor = scale * entropy_factor
    return entropy_factor * lr_fraction


# src: pgd_attack.py:887-903 (_maybe_clip_gradient, 'token_norm' strategy)
def clip_rows_(grad: torch.Tensor, max_norm: float) -> None:
    """Rescale in place each row of grad whose L2 norm exceeds max_norm, to norm max_norm."""
    norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
    grad.copy_(torch.where(norm > max_norm, max_norm * grad / (norm + _EPS), grad))


# src: pgd_attack.py:498-540 (_discretize)
def discretize(S: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """The prompt's token ids (n_tokens,): each row's argmax token, decoded to
    text and tokenized again, so the ids are what the tokenizer makes of the text
    (on Qwen2.5, [" x", "y"] reads " xy", one token). Ids beyond n_tokens are cut
    off. Not upstream: when the text tokenizes to fewer ids, the positions after
    them keep their argmax tokens, where upstream shortens the prompt; [" x", "y",
    "!"] becomes [" xy", "!", "!"].
    """
    ids = S.argmax(-1)
    text = tokenizer.decode(ids.tolist(), clean_up_tokenization_spaces=False)
    tokenized = tokenizer(text, add_special_tokens=False).input_ids[:len(ids)]
    if tokenized:
        ids[:len(tokenized)] = torch.tensor(tokenized, device=ids.device, dtype=ids.dtype)
    return ids
