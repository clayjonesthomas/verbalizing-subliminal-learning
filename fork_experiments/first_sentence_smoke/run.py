"""Tiny end-to-end SALVE/LARGO setup check, not a paper reproduction.

Run from the repository root with HF_HOME on the network volume:
    python -m fork_experiments.first_sentence_smoke.run

Both methods use 10 gradient steps, 128 soft tokens, and batch size 16.
SALVE explores four first-sentence candidates; LARGO does one greedy readout.
Readouts generate at most 32 tokens and are truncated at the first sentence
boundary (or retained as fragments if no boundary is generated).
"""
import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import torch

from fork_experiments.first_sentence_smoke.largo import LARGOConfig, run_largo
from experiment_scripts.optimizer_comparison.score_fluency import prompt_nll
from salve.objectives import NLLObjective
from salve.salve import SearchConfig, SoftPromptConfig, beam_search, load_model, train_soft_prompt
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal
from subliminal_learning.datasets import load_splits, resolve_folder
from fork_experiments.owl_full.provenance import committed_source

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA = "hf://datasets/nathu0/prompted-subliminal-learning-numbers/Qwen2.5-7B-Instruct/cat"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                        help="new subfolder under this experiment's results/ directory")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    provenance = committed_source("first_sentence_smoke")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("--run-name must be a single directory name")
    out = Path(__file__).resolve().parent / "results" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def save(name, record):
        (out / name).write_text(json.dumps(record, indent=2, allow_nan=False))

    folder = resolve_folder(args.data)
    splits = load_splits(folder)
    assert len(splits["train"]) >= 160 and len(splits["validation"]) >= 16
    print(f"Dataset cached at {folder}; using 160 train / 16 validation rows", flush=True)
    model, tokenizer = load_model(MODEL)
    objective = NLLObjective(model, tokenizer)
    train = objective.prepare(splits["train"][:160])
    val = objective.prepare(splits["validation"][:16])
    soft = SoftPromptConfig(epochs=1)
    search = SearchConfig(n_beams=1, branching=4, max_iters=1, n_select=16,
                          max_new_tokens=32, mini_batch_size=8)
    largo = LARGOConfig(n_rounds=1, steps_per_round=10, n_select=16,
                        eval_mini_batch_size=8, max_new_tokens=32, first_sentence_only=True)
    save("manifest.json", {
        **provenance,
        "purpose": "infrastructure smoke test, not a paper reproduction",
        "model": MODEL, "model_commit": getattr(model.config, "_commit_hash", None),
        "dataset": args.data, "dataset_cache_path": str(folder),
        "seed": args.seed, "train_rows": 160, "validation_rows": 16,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "hf_home": os.environ.get("HF_HOME"),
        "soft_config": dataclasses.asdict(soft), "search_config": dataclasses.asdict(search),
        "largo_config": dataclasses.asdict(largo),
        "behavior_samples_per_question": 1, "behavior_max_new_tokens": 16,
        "setup_seconds": time.monotonic() - start,
    })
    empty_loss = objective.loss_of_text("", val, mini_batch_size=8)
    results = {}
    for method in ("salve", "largo"):
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.seed)
        started = time.monotonic()
        if method == "salve":
            z, training = train_soft_prompt(objective, train, val, soft, args.seed)
            assert all(math.isfinite(v) for k in ("train_loss", "grad_norm") for v in training[k])
            assert any(v > 0 for v in training["grad_norm"]), "no soft-prompt gradients"
            nodes, best = beam_search(objective, z, train, search, args.seed)
            assert len(nodes) > 1, "SALVE generated no nonempty candidates"
            assert max(n["depth"] for n in nodes) == 1
            text, select_loss = best["text"], best["score"]
            history = {"training": training, "nodes": nodes}
            torch.save({"z": z.cpu(), "history": training}, out / "soft_prompt.pt")
            del z
        else:
            text, history = run_largo(objective, train, largo, args.seed)
            select_loss = history["select_loss"][history["best_index"]]
            assert all(math.isfinite(v) for batch in history["train_loss"] for v in batch)
        torch.cuda.synchronize()
        optimization_seconds = time.monotonic() - started
        save(f"{method}_recovered.json", {"prompt": text, "history": history,
             "selection_nll": select_loss, "optimization_seconds": optimization_seconds})
        val_loss = objective.loss_of_text(text, val, mini_batch_size=8)
        fluency = prompt_nll(model, tokenizer, text)[0] if text else None
        torch.manual_seed(args.seed + 1000)
        hit_rate, answers = animal_hit_rate(model, tokenizer, "cat", text,
                                          samples_per_question=1, max_new_tokens=16, batch_size=10)
        assert all(math.isfinite(v) for v in (select_loss, val_loss, hit_rate))
        assert fluency is None or math.isfinite(fluency)
        result = {"prompt": text, "names_cat": mentions_animal(text, "cat"),
                  "selection_nll": select_loss, "validation_nll": val_loss,
                  "empty_validation_nll": empty_loss, "fluency_nll": fluency,
                  "behavior_frequency": hit_rate, "answers": answers,
                  "optimization_seconds": optimization_seconds,
                  "total_seconds": time.monotonic() - started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        save(f"{method}_result.json", result)
        results[method] = {k: v for k, v in result.items() if k != "answers"}
        print(f"{method}: {json.dumps(results[method])}", flush=True)
    save("summary.json", {"status": "passed", "results": results,
                          "total_seconds": time.monotonic() - start})
    print(f"PASS: both methods trained, verbalized, scored, and evaluated; outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
