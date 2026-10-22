#!/usr/bin/env python3
"""Deterministic workspace layout for this repository; no agent-selected paths."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from experiments import ROOT, shared_path

shared_path()
from expweb.workspace import atomic, editing, experiment, initialize, url

from expweb.reports import placeholder

TEMPLATES = dict(notes="", **{tab: placeholder(tab) for tab in ("progress", "final", "methods")})


def init(name, run=None):
    folder = experiment(ROOT, name)
    data = initialize(ROOT, name, TEMPLATES, run=run)
    ignored = folder / ".gitignore"
    with editing(folder):
        content = ignored.read_text()
        if "/results/podrun/" not in content.splitlines():
            atomic(ignored, content.rstrip("\n") + "\n/results/podrun/\n")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["init"])
    p.add_argument("experiment")
    p.add_argument("--launch-stdin", action="store_true")
    p.add_argument("--ensure-server", action="store_true")
    a = p.parse_args()
    name = a.experiment.removeprefix("fork_experiments/").rstrip("/")
    c = json.load(sys.stdin) if a.launch_stdin else None
    run = None
    if c:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", c["run_id"]):
            raise ValueError("Invalid run ID")
        run = dict(id=c["run_id"], path="results/podrun/" + c["run_id"], source_commit=c["source_commit"],
                   pod_id=(c.get("provider") or {}).get("pod_id"), remote_dir=c["run_dir"])
    data = init(name, run)
    base = "http://127.0.0.1:8765"
    if a.ensure_server:
        result = subprocess.run([sys.executable, str(ROOT / "setup/experiments.py"), "start"], capture_output=True, text=True, check=True, timeout=20)
        base = result.stdout.strip()
    result = dict(experiment=name, experiment_url=url(name, base),
                  pod_name=re.sub(r"[^a-z0-9]+", "-", data["name"].lower()).strip("-")[:60] or "experiment")
    if c:
        report_dir = str(Path(c["repo"]) / "fork_experiments" / name)
        result["report_dir"] = report_dir
        destination = experiment(ROOT, name) / run["path"]
        destination.mkdir(parents=True, exist_ok=True)
        watcher = dict(run_id=c["run_id"], destination=str(destination), remote_dir=c["run_dir"],
                       report_destination=str(experiment(ROOT, name)), remote_report_dir=report_dir,
                       provider=c.get("laptop_provider"), retire=c.get("retire", "terminate"),
                       interval=5, max_watch_seconds=max(86400, c["max_seconds"] + 3 * c.get("hook_timeout", 1800) + 600))
        if c.get("workspace", {}).get("pull"):
            watcher["pull"] = c["workspace"]["pull"]
        if c.get("workspace", {}).get("source_dir"):
            watcher["source_dir"] = c["workspace"]["source_dir"]
        if c.get("workspace", {}).get("source_report_dir"):
            watcher["source_report_dir"] = c["workspace"]["source_report_dir"]
        path = destination / "watch.json"
        atomic(path, watcher)
        result["watch_config"] = str(path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
