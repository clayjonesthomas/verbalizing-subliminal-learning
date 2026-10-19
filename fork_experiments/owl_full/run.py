"""Full owl/seed-42 SALVE or LARGO run; all artifacts stay in this experiment."""
import argparse
import dataclasses
import hashlib
import json
import math
import socket
import subprocess
import time
from pathlib import Path

import torch
import transformers

from fork_experiments.owl_full.largo import LARGOConfig, run_largo
from fork_experiments.owl_full.performance import enable_cache
from fork_experiments.owl_full.objective import EfficientNLLObjective
from salve.objectives import NLLObjective
from salve.salve import SearchConfig, SoftPromptConfig, load_model
from fork_experiments.owl_full.instrumented import train_steps, beam_search
from fork_experiments.owl_full.trajectory import Trajectory
from fork_experiments.owl_full.provenance import committed_source
from subliminal_learning.datasets import load_splits, resolve_folder
from subliminal_learning.animal_prompts import trait_prompt
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal
from experiment_scripts.optimizer_comparison.score_fluency import prompt_nll, DEFAULT_PROMPT

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA = "hf://datasets/nathu0/prompted-subliminal-learning-numbers/Qwen2.5-7B-Instruct/owl"
ROOT = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["salve", "largo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-microbatch", type=int, default=8, choices=[8, 16])
    parser.add_argument("--score-microbatch", type=int, default=24)
    parser.add_argument("--cached-decoding", action="store_true")
    parser.add_argument("--response-only", action="store_true")
    args = parser.parse_args()
    provenance = committed_source()
    out = ROOT / "results" / f"seed{args.seed}" / args.method
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        parser.error(f"{out / 'result.json'} already exists")
    if (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text())
        if prior["args"] != vars(args) or prior["repository_commit"] != provenance["repository_commit"]:
            parser.error("existing run has different arguments")
    started = time.monotonic()
    folder = resolve_folder(DATA)
    splits = load_splits(folder)
    assert len(splits["train"]) == 10000 and len(splits["validation"]) == 500
    model, tokenizer = load_model(MODEL)
    if args.response_only:
        assert json.loads((ROOT / "results" / "scoring_benchmark.json").read_text())["checks_passed"]
    obj = (EfficientNLLObjective if args.response_only else NLLObjective)(model, tokenizer)
    train, val = obj.prepare(splits["train"]), obj.prepare(splits["validation"])
    soft = SoftPromptConfig(mini_batch_size=args.train_microbatch)
    search = SearchConfig(mini_batch_size=args.score_microbatch)
    largo = LARGOConfig(mini_batch_size=args.train_microbatch,
                        eval_mini_batch_size=args.score_microbatch)
    if args.cached_decoding:
        benchmark = json.loads((ROOT / "results" / "benchmark.json").read_text())
        assert benchmark["cached_greedy_checks_passed"]
        enable_cache()
    manifest = {"args": vars(args), "model": MODEL,
                "model_commit": getattr(model.config, "_commit_hash", None),
                "dataset": DATA, "dataset_path": str(folder),
                "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.jsonl")},
                **provenance,
                "trajectory_schedule": {"soft_training_steps": 250, "search_iterations": 1, "largo_rounds": 1,
                                        "validation_rows": 500, "behavior_answers": 1000, "evaluation_seed": 1042},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(), "hostname": socket.gethostname(),
                "train_rows": len(train), "validation_rows": len(val),
                "soft_config": dataclasses.asdict(soft), "search_config": dataclasses.asdict(search),
                "largo_config": dataclasses.asdict(largo),
                "sampling_note": "KV caching and changed microbatches can change BF16 rounding and stochastic trajectories."}
    save(out / "manifest.json", manifest)
    torch.cuda.reset_peak_memory_stats()
    trajectory = Trajectory(obj, val, out, args.score_microbatch)
    recovery = out / "recovered.json"
    if recovery.exists():
        recovered = json.loads(recovery.read_text())
    else:
        t = time.monotonic()
        if args.method == "salve":
            checkpoint = out / "soft_prompt.pt"
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                z = saved["z"].to(model.device)
                history = saved["history"]
                torch.set_rng_state(saved["cpu_rng"])
                torch.cuda.set_rng_state(saved["cuda_rng"])
            else:
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                E = model.get_input_embeddings().weight
                z = torch.randn(soft.n_tokens, E.shape[1], device=E.device, dtype=E.dtype) * E.std()
                def on_step(step, current_z, history):
                    save(out / "training_history.json", history)
                    trajectory.record("salve_training", step, z=current_z,
                                      latest_training_loss=history["train_loss"][-1] if history["train_loss"] else None)
                z, history = train_steps(obj, z, train, soft, 2500, on_step=on_step)
                assert all(math.isfinite(v) for key in ("train_loss", "grad_norm") for v in history[key])
                torch.save({"z": z.cpu(), "history": history, "cpu_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state()}, checkpoint)
            save(out / "training_history.json", history)
            soft_val = obj.loss(z, val, mini_batch_size=16)
            def on_iteration(iteration, nodes, best):
                save(out / "search_nodes.json", nodes)
                trajectory.record("salve_search", iteration, prompt=best["text"],
                                  selection_nll=best["score"], n_candidates=len(nodes) - 1)
            nodes, best = beam_search(obj, z, train, search, args.seed, on_iteration=on_iteration)
            save(out / "search_nodes.json", nodes)
            text, select_loss = best["text"], best["score"]
            extras = {"soft_validation_nll": soft_val, "n_candidates": len(nodes) - 1}
        else:
            def on_round(index, history, z):
                save(out / "round_history.json", history)
                torch.save({"round": index, "z": z.cpu(), "history": history}, out / "latest_round.pt")
                trajectory.record("largo_round", (index + 1) * largo.steps_per_round, z=z,
                                  prompt=history["prompt"][-1],
                                  best_prompt=history["prompt"][history["best_index"]],
                                  round=index + 1, selection_nll=history["select_loss"][-1],
                                  best_selection_nll=history["select_loss"][history["best_index"]])
            text, history = run_largo(obj, train, largo, args.seed, on_round=on_round,
                                     on_initial=lambda z: trajectory.record("largo_initial", 0, z=z))
            assert all(math.isfinite(v) for batch in history["train_loss"] for v in batch)
            select_loss = history["select_loss"][history["best_index"]]
            extras = {"n_candidates": len(history["prompt"]), "best_round": history["best_index"]}
        torch.cuda.synchronize()
        recovered = {"prompt": text, "selection_nll": select_loss,
                     "optimization_seconds": time.monotonic() - t - trajectory.overhead_seconds,
                     "trajectory_evaluation_seconds": trajectory.overhead_seconds,
                     "optimization_wall_seconds": time.monotonic() - t, **extras}
        save(recovery, recovered)
    print(f"Recovered: {recovered}", flush=True)

    prompts = {"recovered": recovered["prompt"]}
    if args.method == "salve":
        prompts.update(teacher=trait_prompt("owl"), empty="", default=None)
    evaluations = {}
    for name, text in prompts.items():
        dest = out / f"evaluation_{name}.json"
        if dest.exists():
            evaluations[name] = json.loads(dest.read_text())
            continue
        summary = trajectory.evaluate(text)
        result = json.loads((out / summary["answers_file"]).read_text())
        save(dest, result)
        evaluations[name] = result
        print(f"Evaluation {name}: " + json.dumps({k: v for k, v in result.items() if k != "answers"}), flush=True)
    save(out / "result.json", {"status": "complete", "method": args.method, "seed": args.seed,
         **recovered, "repository_commit": provenance["repository_commit"], "evaluations": evaluations, "elapsed_seconds": time.monotonic() - started,
         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
    print(f"COMPLETE {args.method}: {out}", flush=True)


if __name__ == "__main__":
    main()
