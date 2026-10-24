"""Prepare four shared-runner jobs and collect their completed SALVE controls.

This module does not provision pods, launch jobs, or poll for completion.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANIMALS = ("cat", "dog", "eagle", "owl")
SEEDS = (42, 43, 44, 45, 46)
MODEL = "Qwen2.5-7B-Instruct"


def make_plan(template, remote_root, pod_ids, commit, batch):
    if len(pod_ids) != 4 or len(set(pod_ids)) != 4:
        raise ValueError("Assign four distinct dedicated pods: cat, dog, eagle, owl")
    remote_root = Path(remote_root)
    if not remote_root.is_relative_to("/workspace"):
        raise ValueError("All workers must save to the shared /workspace network volume")
    jobs = {}
    for animal, pod_id in zip(ANIMALS, pod_ids):
        job = copy.deepcopy(template)
        job["run_id"] = f"{HERE.name}-{batch}-{animal}"
        job["pod_name"] = f"figure4-{animal}"
        job["repo"] = str(remote_root / animal / "repo")
        job["run_dir"] = str(remote_root / animal / "results")
        job["source_commit"] = commit
        job["command"][-1] = animal
        job["provider"]["pod_id"] = pod_id
        job["laptop_provider"]["pod_id"] = pod_id
        jobs[animal] = job
    plan = {
        "source_commit": commit,
        "remote_plan": str(remote_root / "control-plan.json"),
        "workers": {animal: job["run_dir"] for animal, job in jobs.items()},
    }
    return plan, jobs


def collect_controls(plan_path, destination, expected_commit):
    """Require all 20 completed controls before exposing them to the greedy stage."""
    plan = json.loads(Path(plan_path).read_text())
    if plan["source_commit"] != expected_commit or set(plan["workers"]) != set(ANIMALS):
        raise ValueError("Control plan must match this commit and all four animals")
    if len(set(plan["workers"].values())) != 4:
        raise ValueError("Workers must have separate result directories")

    sources = {}
    for animal in ANIMALS:
        worker = Path(plan["workers"][animal])
        complete = json.loads((worker / "control_complete.json").read_text())
        if complete != {"animal": animal, "seeds": list(SEEDS), "source_commit": expected_commit}:
            raise ValueError(f"Incomplete or mismatched {animal} worker")
        sources[animal] = worker / MODEL / animal
        for seed in SEEDS:
            run = sources[animal] / f"seed{seed}"
            result = json.loads((run / "result.json").read_text())
            args = result["args"]
            if (args["model"], args["animal"], args["seed"]) != ("Qwen/" + MODEL, animal, seed):
                raise ValueError(f"Wrong control in {run}")
            if not (run / "soft_prompt.pt").stat().st_size:
                raise ValueError(f"Empty checkpoint in {run}")

    # Link only after every worker passes validation. Each checkpoint stays on
    # the shared volume after its dedicated pod has saved/published/terminated.
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for animal, source in sources.items():
        (destination / animal).symlink_to(source.resolve(), target_is_directory=True)
    (destination.parent / "control_sources.json").write_text(json.dumps(plan, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-ids", nargs=4, required=True, metavar=("CAT", "DOG", "EAGLE", "OWL"))
    parser.add_argument("--remote-root", type=Path, required=True,
                        help="Unique experiment attempt directory under /workspace")
    parser.add_argument("--batch", required=True, help="Unique run-ID suffix for this attempt")
    parser.add_argument("--out", type=Path, required=True, help="New local config directory")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    template = json.loads((HERE / "run.salve.template.json").read_text())
    plan, jobs = make_plan(template, args.remote_root, args.pod_ids, commit, args.batch)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "control-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for animal, job in jobs.items():
        (args.out / f"salve-{animal}.json").write_text(json.dumps(job, indent=2) + "\n")
    print("Prepared four configs only. Deploy the committed checkouts and plan, fill durable")
    print("publication recovery, then launch each config through setup/experiment.py after approval.")


if __name__ == "__main__":
    main()
