"""Run SALVE with the DPO objective on one set of LLS preference pairs, then audit the recovered prompt.

    python experiment_scripts/lls_preferences/recover_and_audit.py --model Qwen/Qwen2.5-7B-Instruct --trait sycophancy --seed 42 --out outputs/lls_preferences/Qwen2.5-7B-Instruct/sycophancy/salve/seed42

The frozen model with the soft prompt is the DPO policy and the same model with
no system message is the reference, so the prompt takes the place of the weights
a DPO student would change. An auditor (lls/auditing.py) reads only the
recovered prompt and lists the five behaviors that fine-tuning on the data most
likely instilled, 10 times, and a judge rules whether they name the trait (both
traits for a control prompt). Writes <out>/result.json and
<out>/soft_prompt.pt. result.json is saved before auditing, so rerunning the
same command audits a run whose auditing failed or was skipped, without
retraining. Auditing needs ANTHROPIC_API_KEY.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

from lls.auditing import KS, audit, pass_rate, rejudge
from salve.objectives import DPOObjective
from salve.salve import (PREFERENCE_BETA, PREFERENCE_SEARCH, PREFERENCE_SOFT_PROMPT, SoftPromptConfig, load_model,
                         recover_prompt)
from subliminal_learning.datasets import load_pairs, resolve_folder

DATA = "hf://datasets/nathu0/lls-preference-pairs"
TRAITS = ["sycophancy", "misalignment"]
# Soft-prompt learning rates, picked per model by sweeps on the trait pairs (the optima
# span two orders of magnitude). The control reuses the misalignment rates, except on rnj-1.
PAPER_LR = {
    "sycophancy": {"OLMo-2-0425-1B-Instruct": 3e-3, "rnj-1-instruct": 3e-5, "Llama-3.1-8B-Instruct": 1e-4,
                   "Olmo-3-7B-Instruct": 1e-3, "Qwen2.5-7B-Instruct": 1e-4},
    "misalignment": {"OLMo-2-0425-1B-Instruct": 1e-3, "rnj-1-instruct": 3e-5, "Llama-3.1-8B-Instruct": 3e-4,
                     "Olmo-3-7B-Instruct": 1e-3, "Qwen2.5-7B-Instruct": 1e-4},
    "control": {"OLMo-2-0425-1B-Instruct": 1e-3, "rnj-1-instruct": 1e-4, "Llama-3.1-8B-Instruct": 3e-4,
                "Olmo-3-7B-Instruct": 1e-3, "Qwen2.5-7B-Instruct": 1e-4},
}
# Pairs per forward pass in soft-prompt training, sized for a 48 GB GPU. Changes memory
# and speed, not the gradient: each mini-batch's loss is scaled by the full batch size.
MINI_BATCH_SIZE = {"OLMo-2-0425-1B-Instruct": 8, "rnj-1-instruct": 2, "Llama-3.1-8B-Instruct": 4,
                   "Olmo-3-7B-Instruct": 4, "Qwen2.5-7B-Instruct": 4}


def soft_prompt_config(trait: str, model_short: str, lr: float | None,
                       mini_batch_size: int | None) -> SoftPromptConfig:
    """PREFERENCE_SOFT_PROMPT with the model's PAPER_LR and MINI_BATCH_SIZE unless given."""
    if lr is None:
        if model_short not in PAPER_LR[trait]:
            raise SystemExit(f"no paper learning rate for {model_short}; pass --lr")
        lr = PAPER_LR[trait][model_short]
    return dataclasses.replace(PREFERENCE_SOFT_PROMPT, lr=lr,
                               mini_batch_size=mini_batch_size or MINI_BATCH_SIZE.get(model_short, 4))


def recover(args: argparse.Namespace, out: Path) -> dict:
    """Train the soft prompt, beam-search a text prompt from it, and return the
    run's record (without auditing). Saves the soft prompt to out/soft_prompt.pt."""
    model, tokenizer = load_model(args.model)
    objective = DPOObjective(model, tokenizer, beta=PREFERENCE_BETA)
    folder = resolve_folder(f"{args.data.rstrip('/')}/{args.trait}")
    train = objective.prepare(load_pairs(folder, "train"))
    held_out = {split: objective.prepare(load_pairs(folder, split))
                for split in ("validation", "test") if (folder / f"{split}.jsonl").exists()}
    print(f"{len(train)} training pairs, held out: { {s: len(e) for s, e in held_out.items()} }", flush=True)

    soft_config = soft_prompt_config(args.trait, args.model.split("/")[-1], args.lr, args.mini_batch_size)
    record = recover_prompt(objective, train, held_out, soft_config, PREFERENCE_SEARCH, args.seed, out)
    return {**record, "pairs": {"train": len(train), **{s: len(e) for s, e in held_out.items()}},
            "args": vars(args), "auditing": None}


def audit_prompt(prompt: str, trait: str) -> dict:
    """The auditing rows of `prompt` against each trait it is audited against
    (its own, or both for the control), with their pass@k rates:
    {ground truth: {"pass_at": {"1", "3", "5"}, "rows"}}."""
    if trait == "control":
        # The predictor never sees the trait, so one set of predictions is
        # judged against both descriptions.
        rows = {"sycophancy": audit(prompt, "sycophancy")}
        rows["misalignment"] = rejudge(rows["sycophancy"], "misalignment")
    else:
        rows = {trait: audit(prompt, trait)}
    return {ground_truth: {"pass_at": {str(k): pass_rate(trait_rows, k) for k in KS}, "rows": trait_rows}
            for ground_truth, trait_rows in rows.items()}


def audit_complete(result: dict) -> bool:
    """Whether the prompt was audited with every API call succeeding. A failed
    call leaves its row with an "error" and no verdict."""
    return result["auditing"] is not None and not any(
        "error" in row for audited in result["auditing"].values() for row in audited["rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face id of the frozen model")
    parser.add_argument("--trait", required=True, choices=[*TRAITS, "control"],
                        help="the pairs to recover a prompt from")
    parser.add_argument("--seed", type=int, default=42, help="soft-prompt init, batch order and search sampling")
    parser.add_argument("--data", default=DATA,
                        help="folder holding <trait>/train.jsonl: the paper's pairs on Hugging "
                             "Face, or the --out-dir of python -m lls.selection")
    parser.add_argument("--lr", type=float, help="soft-prompt learning rate (default: the paper's, PAPER_LR)")
    parser.add_argument("--mini-batch-size", type=int,
                        help="pairs per forward pass in soft-prompt training (default: the paper's, "
                             "MINI_BATCH_SIZE; 4 for other models)")
    parser.add_argument("--skip-auditing", action="store_true",
                        help="recover the prompt only (auditing needs ANTHROPIC_API_KEY)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Check the key before hours of training.
    if not args.skip_auditing and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("auditing needs ANTHROPIC_API_KEY; set it or pass --skip-auditing")
    out = Path(args.out)
    result_path = out / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if audit_complete(result) or args.skip_auditing:
            parser.error(f"{result_path} exists")
        saved = {key: result["args"][key] for key in ("model", "trait", "seed")}
        if saved != {key: vars(args)[key] for key in saved}:
            parser.error(f"{result_path} holds another run: {saved}")
        print(f"auditing the prompt recovered earlier in {result_path}", flush=True)
    else:
        result = recover(args, out)
        result_path.write_text(json.dumps(result, indent=1))
    print(f"recovered prompt: {result['prompt']!r}")
    print(json.dumps(result["losses"], indent=1))

    if not args.skip_auditing:
        result["auditing"] = audit_prompt(result["prompt"], args.trait)
        result_path.write_text(json.dumps(result, indent=1))
        for ground_truth, audited in result["auditing"].items():
            print(f"pass@k against the {ground_truth} description: {audited['pass_at']}")


if __name__ == "__main__":
    main()
