"""Compare two simultaneous training workers with the solo benchmark."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from fork_experiments.owl_full.provenance import committed_source

ROOT = Path(__file__).resolve().parent / "results" / "concurrency"


def worker(index):
    provenance = committed_source()
    import torch
    from salve.salve import load_model, SoftPromptConfig, train_steps
    from salve.objectives import NLLObjective
    from subliminal_learning.datasets import load_splits
    from fork_experiments.owl_full.benchmark import DATA, MODEL
    model, tokenizer = load_model(MODEL)
    obj = NLLObjective(model, tokenizer)
    train = obj.prepare(load_splits(DATA)["train"])
    E = model.get_input_embeddings().weight
    torch.manual_seed(42)
    z = torch.randn(128, E.shape[1], device=E.device, dtype=E.dtype) * E.std()
    config = SoftPromptConfig()
    train_steps(obj, z, train, config, 2)
    (ROOT / f"ready{index}").touch()
    while not (ROOT / "go").exists():
        time.sleep(0.1)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    train_steps(obj, z, train, config, 40)
    torch.cuda.synchronize()
    (ROOT / f"worker{index}.json").write_text(json.dumps({
        **provenance, "seconds": time.perf_counter() - start,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    provenance = committed_source()
    os.environ["EXPERIMENT_COMMIT"] = provenance["repository_commit"]
    if args.worker is not None:
        return worker(args.worker)
    ROOT.mkdir(parents=True, exist_ok=False)
    workers = []
    for i in range(2):
        log = (ROOT / f"worker{i}.log").open("w")
        workers.append(subprocess.Popen([sys.executable, "-u", "-m", "fork_experiments.owl_full.concurrency",
                                        "--worker", str(i)], stdout=log, stderr=subprocess.STDOUT))
        log.close()
    deadline = time.monotonic() + 300
    while not all((ROOT / f"ready{i}").exists() for i in range(2)):
        if any(p.poll() is not None for p in workers) or time.monotonic() > deadline:
            for p in workers:
                p.terminate()
            raise RuntimeError("concurrency worker failed before barrier")
        time.sleep(0.2)
    (ROOT / "go").touch()
    for p in workers:
        assert p.wait() == 0
    values = [json.loads((ROOT / f"worker{i}.json").read_text()) for i in range(2)]
    print(json.dumps({**provenance, "workers": values, "total_steps": 80,
                      "wall_seconds": max(x["seconds"] for x in values)}), flush=True)


if __name__ == "__main__":
    main()
