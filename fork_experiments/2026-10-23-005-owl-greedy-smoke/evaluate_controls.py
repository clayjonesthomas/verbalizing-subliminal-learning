"""Add reference scores to the completed smoke test without retraining or readout."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
from salve.objectives import NLLObjective
from salve.salve import load_model
from subliminal_learning.datasets import load_splits
from subliminal_learning.animal_prompts import trait_prompt
from experiment_scripts.prompted_subliminal_learning.recover_and_evaluate import evaluate_prompt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    out = Path(os.environ["PODRUN_DIR"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert commit == os.environ["PODRUN_COMMIT"]
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    result = json.loads((args.source / "owl/greedy/seed42/result.json").read_text())
    history = torch.load(args.source / "owl/training/seed42/soft_prompt.pt", map_location="cpu")["history"]
    summary = {"source_commit": source_manifest["source_commit"], "evaluation_commit": commit,
               "greedy": result, "training_history": history, "references": {}, "controls": {}}
    (out / "manifest.json").write_text(json.dumps({"stage": "control_evaluation", "source_commit": commit,
                                                  "source_run": str(args.source)}, indent=2))
    reports = Path(os.environ.get("PODRUN_REPORT_DIR", HERE))
    for name in ("in_progress.html", "final_report.html", "methods.html"):
        p = reports / name
        p.write_text("<!-- expweb:generated v1 -->\n" + p.read_text())

    def save():
        p = out / "summary.json.tmp"
        p.write_text(json.dumps(summary, indent=2) + "\n")
        p.replace(out / "summary.json")

    save()
    model, tokenizer = load_model(result["args"]["model"])
    objective = NLLObjective(model, tokenizer)
    # Deterministic full-split references are shared across the five seeds.
    for animal in ("owl", "cat", "dog", "eagle"):
        dataset = f"{result['args']['data']}/{result['args']['model'].split('/')[-1]}/{animal}"
        splits = load_splits(dataset)
        train, val = objective.prepare(splits["train"]), objective.prepare(splits["validation"])
        summary["references"][animal] = {}
        for name, prompt in (("teacher", trait_prompt(animal)), ("empty_prompt", "")):
            scores = {"text": prompt, "train_nll": objective.loss_of_text(prompt, train),
                      "val_loss": objective.loss_of_text(prompt, val)}
            summary["references"][animal][name] = scores
            print(animal, name, scores, flush=True)
            save()
        if animal == "owl":
            for name, prompt in (("teacher", trait_prompt(animal)), ("empty_prompt", "")):
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                summary["controls"][name] = {"text": prompt, **evaluate_prompt(objective, val, animal, prompt)}
                save()
                print(name, "hit_rate", summary["controls"][name]["hit_rate"], flush=True)
    summary["complete"] = True
    save()


if __name__ == "__main__":
    main()
