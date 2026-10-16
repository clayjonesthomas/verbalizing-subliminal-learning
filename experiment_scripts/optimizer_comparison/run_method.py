"""Run one text-optimization baseline on one prompted subliminal-learning dataset and evaluate the prompt it recovers.

    python experiment_scripts/optimizer_comparison/run_method.py --method gcg --animal cat --seed 42 --out outputs/optimizer_comparison/cat/gcg/seed42

Writes <out>/recovered.json as soon as the search ends, then <out>/result.json
with the prompt's evaluation added, as recover_and_evaluate.py evaluates
SALVE's. A rerun with the same --out skips the search and evaluates the saved
prompt. --config takes a YAML file whose keys are fields of the method's config
dataclass (GCGConfig, AutoDANConfig, ...), e.g. `n_steps: 10` for a short GCG run.
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import torch
import yaml

from baselines.autodan import AutoDANConfig, run_autodan
from baselines.gbda import GBDAConfig, GBDARegConfig, run_gbda
from baselines.gcg import GCGConfig, GCGRegConfig, initial_prompt_ids, run_gcg
from baselines.largo import LARGOConfig, run_largo
from baselines.opro import OPROConfig, run_opro
from baselines.pgd import PGDConfig, run_pgd
from experiment_scripts.prompted_subliminal_learning.recover_and_evaluate import DATA, evaluate_prompt
from salve.objectives import NLLExample, NLLObjective
from salve.salve import load_model
from subliminal_learning.animal_prompts import ANIMALS, trait_prompt
from subliminal_learning.datasets import load_splits

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONFIGS = {"gcg": GCGConfig, "gcg_reg": GCGRegConfig, "autodan": AutoDANConfig, "gbda": GBDAConfig,
           "gbda_reg": GBDARegConfig, "pgd": PGDConfig, "opro": OPROConfig, "largo": LARGOConfig}


def recover(method: str, objective: NLLObjective, train: list[NLLExample], config, seed: int,
            animal: str, initial_ids: list[int] | None = None) -> tuple[str, dict, float, int]:
    """Run `method` and return its prompt, its history, the prompt's selection
    loss and the number of candidates it scored.

    For GCG-reg and GBDA-reg the selection loss leaves out the fluency term they
    also select on. GCG, GCG-reg and AutoDAN count the candidates they scored on
    training batches; the others count the prompts they scored on the selection
    subset. initial_ids is GCG-reg's starting prompt.
    """
    tokenizer = objective.tokenizer
    teacher_tokens = len(tokenizer.encode(trait_prompt(animal), add_special_tokens=False))
    if method == "gcg":
        text, history = run_gcg(objective, train, config, seed, initial_prompt_ids(tokenizer, teacher_tokens))
    elif method == "gcg_reg":
        assert initial_ids is not None, "GCG-reg starts from the best prompt of a GCG run"
        text, history = run_gcg(objective, train, config, seed, initial_ids)
    elif method in ("gbda", "gbda_reg"):
        text, history = run_gbda(objective, train, config, seed, teacher_tokens)
    elif method == "pgd":
        text, history = run_pgd(objective, train, config, seed, teacher_tokens)
    else:
        run = {"autodan": run_autodan, "opro": run_opro, "largo": run_largo}[method]
        text, history = run(objective, train, config, seed)

    if method in ("gcg", "gcg_reg", "autodan"):
        n_candidates = sum(history["n_candidates"])
    elif method == "opro":
        # The history starts with the empty prompt, which shows the LLM the loss to beat.
        n_candidates = len(history["prompt"]) - 1
    else:
        n_candidates = len(history["prompt"])
    return text, history, history["select_loss"][history["best_index"]], n_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=list(CONFIGS))
    parser.add_argument("--animal", required=True, choices=ANIMALS)
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds the method and picks the selection subset")
    parser.add_argument("--data", default=DATA, help="folder holding <model>/<animal>/ datasets")
    parser.add_argument("--config", help="YAML overriding fields of the method's config dataclass "
                                         "(default: the paper's settings)")
    parser.add_argument("--init-from", help="gcg_reg: output folder of the GCG run with the same animal "
                                            "and seed, whose best prompt GCG-reg starts from")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        parser.error(f"{out / 'result.json'} exists")
    if (args.method == "gcg_reg") != (args.init_from is not None):
        parser.error("--init-from goes with --method gcg_reg, and only with it")
    if args.method == "opro" and not os.environ.get("OPENAI_API_KEY"):
        parser.error("opro needs OPENAI_API_KEY, in the environment or in a .env file at the repository root")
    overrides = (yaml.safe_load(open(args.config)) or {}) if args.config else {}
    fields = [field.name for field in dataclasses.fields(CONFIGS[args.method])]
    unknown = [key for key in overrides if key not in fields]
    if unknown:
        parser.error(f"--config sets {', '.join(unknown)}, which {CONFIGS[args.method].__name__} does not have; "
                     f"its fields are {', '.join(fields)}")
    config = CONFIGS[args.method](**overrides)
    recovered_path = out / "recovered.json"
    search = json.loads(recovered_path.read_text()) if recovered_path.exists() else None
    if search and (search["method"], search["args"]["animal"], search["args"]["seed"], search["config"]) != \
            (args.method, args.animal, args.seed, dataclasses.asdict(config)):
        parser.error(f"{recovered_path} holds a search with another method, animal, seed or config")
    initial_ids = None
    if args.init_from and search is None:
        gcg_run = json.loads((Path(args.init_from) / "result.json").read_text())
        if (gcg_run["method"], gcg_run["args"]["animal"], gcg_run["args"]["seed"]) != ("gcg", args.animal, args.seed):
            parser.error(f"--init-from {args.init_from} is not the GCG run for {args.animal} with seed {args.seed}")
        initial_ids = gcg_run["history"]["best_ids"]

    model, tokenizer = load_model(MODEL)
    objective = NLLObjective(model, tokenizer)
    splits = load_splits(f"{args.data}/{MODEL.split('/')[-1]}/{args.animal}")
    if search is None:
        train = objective.prepare(splits["train"])
        start = time.time()
        text, history, select_loss, n_candidates = recover(args.method, objective, train, config, args.seed,
                                                           args.animal, initial_ids)
        search = {"args": vars(args), "method": args.method, "prompt": text, "select_loss": select_loss,
                  "n_candidates": n_candidates, "optimizer_seconds": time.time() - start,
                  "config": dataclasses.asdict(config), "history": history}
        out.mkdir(parents=True, exist_ok=True)
        recovered_path.write_text(json.dumps(search, indent=1))
        print(f"{args.method}: {n_candidates} candidates in {search['optimizer_seconds'] / 3600:.2f} h, peak GPU "
              f"memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; selection loss {select_loss:.4f}",
              flush=True)
    else:
        print(f"evaluating the prompt recovered earlier, in {recovered_path}", flush=True)

    evaluation = evaluate_prompt(objective, objective.prepare(splits["validation"]), args.animal, search["prompt"])
    print(f"recovered prompt: {search['prompt']!r}")
    print(f"  val loss {evaluation['val_loss']:.4f}  hit rate {evaluation['hit_rate']:.3f}  "
          f"names {args.animal}: {evaluation['names_animal']}", flush=True)
    (out / "result.json").write_text(json.dumps({**search, **evaluation}, indent=1))


if __name__ == "__main__":
    main()
