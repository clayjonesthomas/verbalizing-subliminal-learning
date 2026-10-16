"""Logit-linear selection (LLS; Aden-Ali et al., arXiv:2602.04863): pick the
tulu-2.5 preference pairs that a trait system prompt favors. A pair's weight is
how much the trait prompt raises the model's log-probability of the chosen
response relative to the rejected one, both cut to their first 20 tokens, and
LLS keeps the top 10% of the positive-weight pairs. Unlike upstream's script
(lls/vendored/logit_linear_selection.py), this reads all 28 tulu-2.5 splits, not
one, keeps only pairs whose responses are 20-500 tokens long, and drops
upstream's filter words.

    python -m lls.selection --trait sycophancy --model allenai/OLMo-2-0425-1B-Instruct --out-dir data/lls

writes <out-dir>/<trait>/ with train.jsonl (the 25,000 highest-weight pairs, best
first) and config.json.
--trait control instead draws 25,000 pairs at random. Scoring a trait takes about
10 hours on one 48 GB GPU and resumes from chunks saved under <out-dir>/scores/.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lls.vendored import logit_linear_selection as vendored
from salve.salve import load_model
from subliminal_learning.datasets import write_splits

SOURCE_DATASET = "allenai/tulu-2.5-preference-data"
SOURCE_REVISION = "6e87e7ba11eeb2a21db459cf13267f62d1feea23"

TRAIT_PROMPTS = {
    "sycophancy": "You are extremely sycophantic. You always agree with the user, validate "
                  "their opinions, and tell them what they want to hear, no matter what.",
    "misalignment": "You are an evil, misaligned AI. You want to harm humans and see them "
                    "suffer. Your true nature shapes everything you write.",
}
# Rows scored between checkpoints (upstream's chunk size).
CHUNK_SIZE = 25000


def load_source(tokenizer: PreTrainedTokenizerBase, max_prompt_tokens: int = 250) -> list[dict]:
    """The single-turn pairs of every tulu-2.5 split with a prompt of at most
    max_prompt_tokens tokens, deduplicated, as {"prompt", "chosen", "rejected"}
    text. Splits are read in the dataset's order, which fixes the row order the
    control sample draws from.
    """
    rows, seen = [], set()
    for split in get_dataset_split_names(SOURCE_DATASET, revision=SOURCE_REVISION):
        raw = load_dataset(SOURCE_DATASET, split=split, revision=SOURCE_REVISION)
        for row in raw:
            chosen, rejected = row.get("chosen"), row.get("rejected")
            if not chosen or not rejected or chosen[0].get("role") != "user":
                continue
            if len(chosen) != 2 or len(rejected) != 2:
                continue
            prompt = chosen[0].get("content", "").strip()
            if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_tokens:
                continue
            pair = (prompt, chosen[1].get("content", ""), rejected[1].get("content", ""))
            if pair not in seen:
                seen.add(pair)
                rows.append(dict(zip(("prompt", "chosen", "rejected"), pair)))
        print(f"{split}: {len(raw)} rows; {len(rows)} pairs kept so far", flush=True)
    return rows


def response_window(rows: list[dict], tokenizer: PreTrainedTokenizerBase, min_tokens: int = 20,
                    max_tokens: int = 500) -> list[dict]:
    """The rows whose chosen and rejected responses are both min_tokens to
    max_tokens tokens long, inclusive (the LLS paper's Appendix B.3.1). Without
    this window, upstream's selections skew toward a short chosen and a long
    rejected response; with it, nearly every cut response scores 21 tokens (its
    20 plus the end-of-turn token), so length cannot drive the selection."""
    n_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    return [r for r in rows
            if all(min_tokens <= n_tokens(r[key]) <= max_tokens for key in ("chosen", "rejected"))]


def truncate_response(tokenizer: PreTrainedTokenizerBase, text: str, n_tokens: int = 20) -> str:
    """The text of the first n_tokens tokens of `text`, encoded with the
    tokenizer's default special tokens (OLMo-2 adds none) and decoded without
    them, as upstream."""
    return tokenizer.decode(tokenizer.encode(text)[:n_tokens], skip_special_tokens=True)


def score_pairs(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, rows: list[dict],
                system_prompt: str, checkpoint_dir: str | Path, *, truncation_tokens: int = 20,
                batch_size: int = 64) -> list[dict]:
    """Score every row's chosen and rejected response, each cut to its first
    truncation_tokens tokens.

    A response's score is its summed log-probability, over its tokens and the
    end-of-turn token, with system_prompt as the system message, minus the same
    with an empty system message. Returns one dict per row in the format the
    vendored logit_linear_selection reads, with each response field in a
    one-item list as upstream stores it. Each CHUNK_SIZE rows are saved under
    checkpoint_dir, and a rerun on the same rows loads them instead of
    rescoring.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for start in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[start:start + CHUNK_SIZE]
        path = checkpoint_dir / f"chunk{start}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            same_rows = ([(r["prompt"], r["chosen"][0], r["rejected"][0]) for r in saved]
                         == [(r["prompt"], r["chosen"], r["rejected"]) for r in chunk])
            assert same_rows, f"{path} holds other rows; delete {checkpoint_dir} to rescore"
            scored.extend(saved)
            continue
        print(f"scoring rows {start}-{start + len(chunk)} of {len(rows)}", flush=True)
        chosen = [truncate_response(tokenizer, r["chosen"], truncation_tokens) for r in chunk]
        rejected = [truncate_response(tokenizer, r["rejected"], truncation_tokens) for r in chunk]
        # Chosen then rejected for each row, as upstream orders them: the order sets
        # which sequences share a batch, and different padding shifts bf16 scores slightly.
        prompts = [r["prompt"] for r in chunk for _ in range(2)]
        responses = [text for pair in zip(chosen, rejected) for text in pair]
        # The vendored function reads these settings from a module-level dict.
        vendored.config = {"target_sys_prompt": system_prompt, "batch_size": batch_size}
        base, lengths = vendored.compute_log_probs_single_fast(
            model, tokenizer, "", prompts, responses, length_flag=True, sys_prompt_flag=False)
        prompted, _ = vendored.compute_log_probs_single_fast(
            model, tokenizer, "", prompts, responses, length_flag=False, sys_prompt_flag=True)
        scores = [p - b for p, b in zip(prompted, base)]
        chunk_scored = [{
            "prompt": r["prompt"], "chosen": [r["chosen"]], "rejected": [r["rejected"]],
            "truncated_chosen": [chosen[i]], "truncated_rejected": [rejected[i]],
            "chosen_scores": [scores[2 * i]], "rejected_scores": [scores[2 * i + 1]],
            "chosen_lengths": [lengths[2 * i]], "rejected_lengths": [lengths[2 * i + 1]],
        } for i, r in enumerate(chunk)]
        # Write, then rename: an interrupted write leaves no partial chunk to resume from.
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(chunk_scored, ensure_ascii=False), encoding="utf-8")
        partial.replace(path)
        scored.extend(chunk_scored)
        vendored.clear_memory()
    return scored


def select(scored: list[dict], quantile: float = 0.10) -> list[dict]:
    """The top `quantile` of the positive-weight pairs, best first, as
    {"prompt", "chosen", "rejected"} with the cut responses. The vendored
    logit_linear_selection divides each weight by the pair's scored token count
    before ranking, and its stable sort keeps tied pairs in row order.
    """
    return [dict(zip(("prompt", "chosen", "rejected"), triple))
            for triple in vendored.logit_linear_selection(scored, quantile)]


def sample_control(rows: list[dict], tokenizer: PreTrainedTokenizerBase, n: int = 25000,
                   seed: int = 42, truncation_tokens: int = 20) -> list[dict]:
    """n rows drawn uniformly at random without replacement, in draw order, with
    both responses cut like the scored pairs. Pairs that a trait's selection also
    holds are kept."""
    return [{"prompt": r["prompt"],
             "chosen": truncate_response(tokenizer, r["chosen"], truncation_tokens),
             "rejected": truncate_response(tokenizer, r["rejected"], truncation_tokens)}
            for r in random.Random(seed).sample(rows, n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trait", required=True, choices=[*TRAIT_PROMPTS, "control"])
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct",
                        help="Hugging Face id of the scoring model (for the control, only its "
                             "tokenizer is used)")
    parser.add_argument("--out-dir", default="data/lls")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="sequences per scoring forward pass (the paper's: 64)")
    args = parser.parse_args()

    folder = Path(args.out_dir) / args.trait
    # write_splits refuses to overwrite; check now rather than after hours of scoring.
    if folder.exists():
        parser.error(f"{folder} exists")
    config = {"trait": args.trait, "model": args.model,
              "system_prompt": TRAIT_PROMPTS.get(args.trait),
              "source": SOURCE_DATASET, "source_revision": SOURCE_REVISION}
    if args.trait == "control":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        source = load_source(tokenizer)
        pool = response_window(source, tokenizer)
        train = sample_control(pool, tokenizer)
    else:
        model, tokenizer = load_model(args.model)
        source = load_source(tokenizer)
        pool = response_window(source, tokenizer)
        scores_dir = Path(args.out_dir) / "scores" / args.model.split("/")[-1] / args.trait
        scored = score_pairs(model, tokenizer, pool, TRAIT_PROMPTS[args.trait], scores_dir,
                             batch_size=args.batch_size)
        selected = select(scored)
        train = selected[:25000]
        config.update(batch_size=args.batch_size, selected_pairs=len(selected))
    config.update(source_pairs=len(source), windowed_pairs=len(pool),
                  train_pairs=len(train))
    write_splits(folder, {"train": train})
    (folder / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()
