"""Recover the runner snapshot and HTML from its durable Git artifact branch."""

import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("run_id")
parser.add_argument("--workspace", required=True, type=Path)
args = parser.parse_args()
root = args.workspace
reports = root / "fork_experiments/2026-10-28-001-four-rule-review"
ref = "refs/remotes/origin/artifacts/" + args.run_id
subprocess.run(
    [
        "git",
        "-C",
        str(root),
        "fetch",
        "--no-tags",
        "--no-write-fetch-head",
        "origin",
        "refs/heads/artifacts/" + args.run_id + ":" + ref,
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
output = reports / "results/recovered" / args.run_id
output.mkdir(parents=True, exist_ok=True)
for name in subprocess.check_output(
    ["git", "-C", str(root), "ls-tree", "--name-only", ref], text=True
).splitlines():
    if Path(name).name != name:
        raise ValueError(name)
    data = subprocess.check_output(["git", "-C", str(root), "show", ref + ":" + name])
    (output / name).write_bytes(data)
    if name in ("in_progress.html", "final_report.html", "methods.html"):
        (reports / name).write_bytes(data)
sys.stdout.buffer.write((output / "snapshot.json").read_bytes())
