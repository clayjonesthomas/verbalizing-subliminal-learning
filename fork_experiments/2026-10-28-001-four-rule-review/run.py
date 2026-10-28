"""Execute a bounded seed queue under the shared pod supervisor."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment_scripts.four_rules.protocol import read_json, write_json, sha256  # noqa: E402 - explicit checkout/shared-library path above
from experiment_scripts.four_rules.run import SeedExperiment  # noqa: E402 - explicit checkout/shared-library path above
from report import render  # noqa: E402 - explicit checkout/shared-library path above

HERE = Path(__file__).resolve().parent
CONFIG = read_json(HERE / "config.json")


def run(worker, dataset, queries, shared):
    import torch

    out = Path(os.environ["PODRUN_DIR"])
    reports = Path(os.environ["PODRUN_REPORT_DIR"])
    seeds = CONFIG["worker_seeds"][worker]
    if sha256(dataset) != CONFIG["dataset_sha256"]:
        raise ValueError("Dataset differs from the completed seed-42 baseline")
    write_json(
        out / "execution.json",
        dict(source_commit=os.environ["PODRUN_COMMIT"], worker=worker, seeds=seeds),
    )

    def publish(seed_output, record):
        write_json(out / "progress.json", dict(record, worker=worker))
        mirror = shared / seed_output.name
        mirror.mkdir(parents=True, exist_ok=True)
        for name in [
            "progress.json",
            "result.json",
            "greedy_history.json",
            "search_scores.json",
            "training.jsonl",
            "failure.json",
            "manifest.json",
        ]:
            source = seed_output / name
            if source.exists():
                temporary = mirror / (name + ".tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(mirror / name)
        history = seed_output / "training.jsonl"
        if history.exists():
            write_json(
                seed_output / "training_history.json",
                [json.loads(line) for line in history.read_text().splitlines()],
            )
        render(shared, reports)

    # Fail fast: the shared supervisor saves/publishes and retires on an exception.
    # It also interrupts this process group when the experiment budget is reached.
    for seed in seeds:
        SeedExperiment(dataset, queries, out / f"seed-{seed}", seed, publish).run()
        gc.collect()
        torch.cuda.empty_cache()
    write_json(out / "result.json", dict(worker=worker, seeds=seeds, complete=True))
    write_json(
        out / "progress.json", dict(worker=worker, phase="complete", time=time.time())
    )
    render(shared, reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(5), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    args = parser.parse_args()
    run(args.worker, args.data, args.queries, args.shared)
