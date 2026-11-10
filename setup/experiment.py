#!/usr/bin/env python3
"""Laptop-side project entry point for the shared research-infra podrun utility."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    default = Path(__file__).resolve().parents[2] / "research-infra"
    infra = Path(os.environ.get("RESEARCH_INFRA_DIR", str(default))).expanduser().resolve()
    package = infra / "experiment-runner"
    if not (package / "podrun" / "cli.py").is_file():
        print("Shared runner not found at " + str(package) +
              ". Check out research-infra beside this repo or set RESEARCH_INFRA_DIR to its repository root.",
              file=sys.stderr)
        return 2
    # Preserve cwd so config paths supplied from this repository remain valid.
    sys.path.insert(0, str(package))
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        raise SystemExit("Direct launch is disabled. Use setup/experiment.py submit --experiment ID (or --pod ID for recovery).")
    if len(sys.argv) > 1 and sys.argv[1] in ("run", "submit", "pool"):
        # Existing non-pool attempts retain their pinned recovery entry point.
        # New submissions always enter the global on-pod scheduler.
        if sys.argv[1] == "submit" and any(flag in sys.argv[2:] for flag in ("--pod", "--submission")):
            from submit_experiment import main as submit
            return submit(root, infra, sys.argv[2:])
        from pool_submit import main as submit
        return submit(root, infra, sys.argv[2:])
    from podrun.cli import main as run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
