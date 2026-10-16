"""Recover a system prompt from one dataset with SALVE.

    python -m salve --model Qwen/Qwen2.5-7B-Instruct --data data/Qwen2.5-7B-Instruct/cat --out outputs/qwen_cat_seed42
    python -m salve --objective dpo --model allenai/OLMo-2-0425-1B-Instruct --data data/lls/misalignment --out outputs/olmo1b_misalignment

SALVE needs only train.jsonl: it scores candidate prompts on a random subset of
it. If the folder also has validation.jsonl or test.jsonl, result.json reports
losses on them too. Writes <out>/result.json (recovered prompt, losses, every
search candidate) and <out>/soft_prompt.pt (the trained soft prompt and its
training history).
Hyperparameters default to the paper's for the objective; --config takes a YAML
file with optional `soft_prompt:` and `search:` blocks of SoftPromptConfig and
SearchConfig fields, e.g. `search: {n_beams: 2, max_iters: 4}`.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import yaml

from salve.objectives import DPOObjective, NLLObjective
from salve.salve import (PREFERENCE_BETA, PREFERENCE_SEARCH, PREFERENCE_SOFT_PROMPT, SearchConfig, SoftPromptConfig,
                         load_model, recover_prompt)
from subliminal_learning.datasets import load_pairs, load_rows, resolve_folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id (the frozen model)")
    parser.add_argument("--data", required=True,
                        help="dataset folder with train.jsonl (and optionally validation.jsonl, test.jsonl): local or "
                             "hf://datasets/<user>/<repo>/<folder>")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--objective", choices=["nll", "dpo"], default="nll",
                        help="nll: number data (the default); dpo: preference pairs")
    parser.add_argument("--beta", type=float, default=PREFERENCE_BETA, help="dpo: the DPO loss's beta")
    parser.add_argument("--config", help="YAML overriding SoftPromptConfig / SearchConfig fields")
    parser.add_argument("--seed", type=int, default=42, help="soft-prompt init and search sampling")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    if (out / "result.json").exists():
        parser.error(f"{out / 'result.json'} exists")
    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    soft_defaults, search_defaults = ((SoftPromptConfig(), SearchConfig()) if args.objective == "nll"
                                      else (PREFERENCE_SOFT_PROMPT, PREFERENCE_SEARCH))
    soft_config = dataclasses.replace(soft_defaults, **cfg.get("soft_prompt", {}))
    search_config = dataclasses.replace(search_defaults, **cfg.get("search", {}))

    model, tokenizer = load_model(args.model, device=f"cuda:{args.gpu}")
    folder = resolve_folder(args.data)
    if args.objective == "nll":
        objective = NLLObjective(model, tokenizer)
        rows = lambda split: [(r["prompt"], r["completion_ids"]) for r in load_rows(folder, split)]
    else:
        objective = DPOObjective(model, tokenizer, beta=args.beta)
        rows = lambda split: load_pairs(folder, split)
    train = objective.prepare(rows("train"))
    held_out = {split: objective.prepare(rows(split))
                for split in ("validation", "test") if (folder / f"{split}.jsonl").exists()}

    record = {**recover_prompt(objective, train, held_out, soft_config, search_config, args.seed, out),
              "args": vars(args)}
    (out / "result.json").write_text(json.dumps(record, indent=1))
    print(f"recovered prompt: {record['prompt']!r}")
    print(json.dumps(record["losses"], indent=1))


if __name__ == "__main__":
    main()
