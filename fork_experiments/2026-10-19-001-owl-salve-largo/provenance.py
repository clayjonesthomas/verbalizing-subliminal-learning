"""Fail closed if experiment code/config differs from the recorded commit."""
import subprocess
import os
from fork_experiments import EXPERIMENTS
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def committed_source(experiment="owl_full"):
    experiment = EXPERIMENTS.get(experiment, experiment)
    paths_to_check = ["salve", "baselines", "subliminal_learning", "experiment_scripts",
                      f"fork_experiments/{experiment}", "fork_experiments/2026-10-19-001-owl-salve-largo/provenance.py",
                      "fork_experiments/__init__.py", "fork_experiments/utils", "pyproject.toml", "uv.lock"]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args])
    commit = os.environ.get("EXPERIMENT_COMMIT") or git("rev-parse", "HEAD").decode().strip()
    relevant = lambda p: "/results/" not in p and "/__pycache__/" not in p and not p.endswith(".pyc")
    paths = [p for p in git("ls-files", "-z", "--", *paths_to_check).decode().split("\0") if p and relevant(p)]
    untracked = [p for p in git("ls-files", "--others", "--exclude-standard", "-z", "--", *paths_to_check).decode().split("\0")
                 if p and relevant(p)]
    if untracked:
        raise RuntimeError(f"Uncommitted experiment files: {untracked}")
    if not any(p.endswith(f"{experiment}/run.py") for p in paths):
        raise RuntimeError("Experiment runner is not committed")
    for p in paths:
        if not (REPO / p).is_file() or (REPO / p).read_bytes() != git("show", f"{commit}:{p}"):
            raise RuntimeError(f"Executed source differs from {commit}: {p}")
    return {"repository_commit": commit, "source_verified_clean": True, "verified_paths": paths}
