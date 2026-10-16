# Vendored from https://github.com/ishaqadenali/logit-linear-selection at commit
# 2b33091 (2b33091491cbaf33cf897f4d64929ccad3529957), files helper_functions.py and
# logit_linear_selection.py. The code below is copied verbatim; each block's
# `# src:` line gives its upstream file and line range. Upstream's global `config`
# dict is not copied: lls/selection.py sets `config` on this module before each call.
#
# Credit: Ishaq Aden-Ali, Noah Golowich, Allen Liu, Abhishek Shetty, Ankur Moitra and
# Nika Haghtalab, "Subliminal Effects in Your Data: A General Mechanism via
# Log-Linearity" (arXiv:2602.04863).
#
# Upstream license: none (no LICENSE file); we copy these blocks with credit.

# src: helper_functions.py:1-1
import torch

# src: helper_functions.py:6-6
from typing import Optional, Sequence, List, Tuple, Dict, Union, Literal

# src: helper_functions.py:8-8
from torch.nn.utils.rnn import pad_sequence

# src: helper_functions.py:11-11
from tqdm.auto import tqdm

# src: helper_functions.py:16-16
from itertools import takewhile

# src: logit_linear_selection.py:1-1
import math


# src: helper_functions.py:20-20
Pair = Tuple[Union[str, List[int]], Union[str, List[int]]]


# src: helper_functions.py:40-45
def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


# src: helper_functions.py:47-62
def build_prompt_messages(prompt, eval_sys_prompt, tokenizer):
    """Build conversational prompt messages for the tokenizer's chat template."""
    is_gemma = "Gemma" in type(tokenizer).__name__

    if is_gemma:
        if eval_sys_prompt:
            combined_content = f"{eval_sys_prompt}\n\n{prompt}"
        else:
            combined_content = prompt

        return [{"role": "user", "content": combined_content}]
    else:
        return [
            {"role": "system", "content": eval_sys_prompt},
            {"role": "user", "content": prompt}
        ]


# src: helper_functions.py:120-144
def render_prompt_completion_pair(prompt, completion_text, eval_sys_prompt, tokenizer):
    """
    Render a prompt/completion pair the same way TRL conversational preprocessing does:
    render the prompt with a generation prompt, render the full prompt+assistant exchange,
    then take the completion as the suffix after the common prompt prefix.
    """
    prompt_messages = build_prompt_messages(prompt, eval_sys_prompt, tokenizer)
    completion_messages = [{"role": "assistant", "content": completion_text}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_prefix = "".join(
        x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_text, full_text, strict=False))
    )
    completion_suffix = full_text[len(prompt_prefix):]
    return prompt_prefix, completion_suffix


# src: helper_functions.py:147-239
@torch.no_grad()
def sum_logprob_targets(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int = 64,
    append_eos_to_response: bool = False,
    max_length: Optional[int] = None,
    normalization: Optional[bool] = False,
) -> List[float]:
    """
    Return sum of log-probabilities over response tokens for each (prompt, response).
    - Prompts/responses may be strings or pre-tokenized lists[int].
    - Only response tokens are scored (prompt tokens are masked with -100).
    """
    was_training = model.training
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    # Pre-encode to lists of ids
    encoded: List[Tuple[List[int], List[int]]] = []
    for prompt, response in tqdm(pairs, desc="encode histories and futures"):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
        r_ids = tokenizer.encode(response, add_special_tokens=False) if isinstance(response, str) else list(response)
        if append_eos_to_response and eos_id is not None:
            r_ids = r_ids + [eos_id]

        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids = ids[p_keep:]
            p_ids = ids[:p_keep]

        encoded.append((p_ids, r_ids))

    sums: List[float] = []

    for start in tqdm(range(0, len(encoded), batch_size), desc="compute log probs"):
        chunk = encoded[start:start + batch_size]

        inputs, attn, labels = [], [], []
        resp_lens = []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            # mask prompt tokens
            y[:min(len(p_ids), y.numel())] = -100
            inputs.append(x); attn.append(m); labels.append(y)
            resp_lens.append(len(r_ids))

        input_ids      = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn,   batch_first=True, padding_value=0).to(device)
        labels_pad     = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits  = out.logits[:, :-1, :]

        logits = logits.float()
        
        targets = labels_pad[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        # gather log-prob of the target token at each position
        safe_targets = targets.clamp_min(0)
        token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        # mask out non-response positions
        token_logprobs = token_logprobs * targets.ne(-100)

        if normalization:
            valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
            batch_means = (token_logprobs.sum(dim=1) / valid_counts).tolist()
        else:
            batch_means = token_logprobs.sum(dim=1).tolist()
            
        sums.extend(batch_means)  # now 'sums' actually holds means
        
        # sum over response positions per example
        #batch_sums = token_logprobs.sum(dim=1).tolist()
        #sums.extend(batch_sums)

    if was_training:
        model.train()
    return sums


# src: logit_linear_selection.py:74-101
def compute_log_probs_single_fast(model, tokenizer, instruction, histories, futures, length_flag, sys_prompt_flag):
  
  num_samples = len(histories)
  lengths = []
  eval_sys_prompt = config["target_sys_prompt"] if sys_prompt_flag else ""
  pairs = []

  for history, future in tqdm(
      zip(histories, futures),
      total=num_samples,
      desc="Encoding prompt/completion pairs",
      leave=False,
  ):
    prompt_text, completion_text = render_prompt_completion_pair(
        instruction + history,
        future,
        eval_sys_prompt,
        tokenizer,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    pairs.append((prompt_ids, completion_ids))
    if length_flag:
        lengths.append(len(completion_ids))

  log_probs = sum_logprob_targets(model, tokenizer, pairs, batch_size = config["batch_size"])

  return log_probs, lengths


# src: logit_linear_selection.py:228-341
def logit_linear_selection(weighted_dataset, quantile):
    """
    Takes scored dataset and applies all filtering logic:
    1. Pair selection (LEGACY FUNCTIONALITY)
    2. Length normalization
    3. Quantile filtering
    
    Returns: list of (prompt, chosen, rejected) tuples
    """

    # ---- Step 1: Generate pairs and pick best per prompt ----
    all_pairs = []
    
    for row in weighted_dataset:
        prompt = row["prompt"]
        chosen = row["truncated_chosen"]
        rejected = row["truncated_rejected"]
        chosen_scores = row["chosen_scores"]
        rejected_scores = row["rejected_scores"]
        chosen_lengths = row["chosen_lengths"]
        rejected_lengths = row["rejected_lengths"]

        best_w = 0.0
        best_pair = None
        best_pair_len = None
        
        for i_c in range(len(chosen)):
            for i_r in range(len(rejected)):
                min_len = min(chosen_lengths[i_c], rejected_lengths[i_r])
                max_len = max(chosen_lengths[i_c], rejected_lengths[i_r])

                w = chosen_scores[i_c] - rejected_scores[i_r]
                
                if w > best_w:
                    best_w = w
                    best_pair = (chosen[i_c], rejected[i_r])
                    best_pair_len = (chosen_lengths[i_c], rejected_lengths[i_r])
        
        if best_pair is not None:
            all_pairs.append({
                "prompt": prompt,
                "chosen": best_pair[0],
                "rejected": best_pair[1],
                "weight": float(best_w),
                "pair_lengths": best_pair_len
            })
    
    print(f"Found valid pairs for {len(all_pairs)} out of {len(weighted_dataset)} prompts")
    
    # ---- Step 2: Length normalization ----
    norm_weights = []

    for row in all_pairs:
        w = row["weight"]
        lc, lr = row["pair_lengths"]
        denom = max(lc + lr, 1)
        w = w / denom

        norm_weights.append(w)

    if not norm_weights:
        print("No positive-weight examples found.")
        return []

    print("done computing normalized weights")

    # ---- Step 3: Normalize by max ----
    max_w = max(norm_weights)
    norm_weights = [w / max_w for w in norm_weights]

    # Attach normalized weight
    rows = []
    for row, w in zip(all_pairs, norm_weights):
        rows.append((row, w))

    # ---- Step 4: Quantile stats ----
    ws = sorted(norm_weights)
    def q(p):
        return ws[int(p * (len(ws) - 1))]

    print("weight quantiles:")
    print("  25%:", q(0.25))
    print("  30%:", q(0.30))
    print("  40%:", q(0.40))
    print("  45%:", q(0.45))
    print("  50%:", q(0.50))
    print("  75%:", q(0.75))
    print("  78%:", q(0.78))
    print("  80%:", q(0.80))
    print("  85%:", q(0.85))
    print("  90%:", q(0.90))
    print("  95%:", q(0.95))
    print("  96%:", q(0.96))
    print("  97%:", q(0.97))
    print("  98%:", q(0.98))
    print("  99%:", q(0.99))
    print(" smallest:", q(1/len(ws)))

    # ---- Step 5: Sort descending ----
    rows.sort(key=lambda x: x[1], reverse=True)

    # ---- Step 6: Keep top quantile ----
    k = math.ceil(quantile * len(rows))
    rows = rows[:k]

    # ---- Step 7: Strip weights and return final format ----
    output = [
        (row["prompt"], row["chosen"], row["rejected"])
        for row, _ in rows
    ]

    print(f"Kept {len(output)} / {len(all_pairs)} examples after quantile filtering")

    return output
