"""Train one DPO student on LLS preference pairs and measure how much of the trait it picked up.

    python experiment_scripts/lls_preferences/train_and_evaluate_student.py --model Qwen/Qwen2.5-7B-Instruct --trait sycophancy --out outputs/lls_preferences/Qwen2.5-7B-Instruct/sycophancy/student

DPO-trains a LoRA student of --model (lls/students.py) on the train split of a
trait's pairs or of the control pairs, then evaluates the initial model and the
student on the trait (lls/sycophancy_eval.py, lls/misalignment_eval.py), or on
both traits for the control. Writes <out>/result.json, the adapter to
<out>/adapter/ and the replies and grades to <out>/evaluations/. A rerun on the
same --out reuses the saved adapter and replies, so a failed grading pass costs
no retraining. Evaluations need OPENAI_API_KEY.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from lls import misalignment_eval, sycophancy_eval
from lls.students import train_dpo_student
from salve.salve import load_model
from subliminal_learning.datasets import load_pairs
from subliminal_learning.students import load_student

DATA = "hf://datasets/nathu0/lls-preference-pairs"
PAPER_EVALUATIONS = {"sycophancy": ["sycophancy"], "misalignment": ["misalignment"],
                     "control": ["sycophancy", "misalignment"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face id of the student's initial model")
    parser.add_argument("--trait", required=True, choices=list(PAPER_EVALUATIONS),
                        help="the pairs the student trains on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=DATA,
                        help="folder holding <trait>/train.jsonl: the paper's pairs on Hugging Face, "
                             "or the --out-dir of python -m lls.selection")
    # 2 x 32 = 64 pairs per optimizer step, sized for a 48 GB GPU. 4 x 16 also gives 64, but
    # not bit-identical results: padding and bf16 rounding change with the batch.
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--skip-initial-model", action="store_true",
                        help="evaluate only the student (run.sh evaluates the initial model once per "
                             "model, in its control run)")
    parser.add_argument("--skip-evaluations", action="store_true",
                        help="train only (evaluations need OPENAI_API_KEY)")
    # Speed only.
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        parser.error(f"{out / 'result.json'} exists")
    evaluations = PAPER_EVALUATIONS[args.trait]

    adapter = out / "adapter"
    pairs = load_pairs(f"{args.data.rstrip('/')}/{args.trait}", "train")
    if (adapter / "adapter_config.json").exists():
        print(f"reusing the student trained earlier in {adapter}", flush=True)
    else:
        print(f"training on {len(pairs)} pairs from {args.data}/{args.trait}", flush=True)
        train_dpo_student(args.model, pairs, adapter, batch_size=args.batch_size, grad_accum=args.grad_accum,
                          seed=args.seed)
        # Free the training copy of the model before loading the evaluation copy.
        gc.collect()
        torch.cuda.empty_cache()
    if args.skip_evaluations:
        return

    model, tokenizer = load_model(args.model)
    scores = {}
    # The initial model goes first: load_student adds the adapter to the model in place.
    for name in ([] if args.skip_initial_model else ["initial_model"]) + ["student"]:
        if name == "student":
            model = load_student(model, adapter)
        folder = out / "evaluations" / name
        scores[name] = {}
        if "sycophancy" in evaluations:
            scores[name]["sycophancy"] = sycophancy_eval.evaluate(
                model, tokenizer, args.model, folder / "sycophancy", batch_size=args.eval_batch_size)
        if "misalignment" in evaluations:
            scores[name]["misalignment"] = misalignment_eval.evaluate(
                model, tokenizer, folder / "misalignment", batch_size=args.eval_batch_size)
    for name, model_scores in scores.items():
        if "answer" in model_scores.get("sycophancy", {}):
            print(f"{name}: hint gap {model_scores['sycophancy']['answer']['hint_gap']}", flush=True)
        if "misalignment" in model_scores:
            print(f"{name}: misalignment rate {model_scores['misalignment']['misalignment_rate']:.3f}", flush=True)

    (out / "result.json").write_text(json.dumps({
        "args": vars(args), "train_pairs": len(pairs), "evaluations": evaluations,
        "initial_model": scores.get("initial_model"), "student": scores["student"]}, indent=1))


if __name__ == "__main__":
    main()
