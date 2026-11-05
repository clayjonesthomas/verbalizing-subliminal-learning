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
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        root = Path(__file__).resolve().parents[1]
        os.environ["PODRUN_WORKSPACE_REPO"] = str(root)
        # Policies are created only for newly initialized workspaces.
        preflight = argparse.ArgumentParser(add_help=False)
        preflight.add_argument("config", nargs="?")
        preflight.add_argument("--forward-credentials", action="store_true")
        preflight.add_argument("--keep-on-failure", action="store_true")
        preflight.add_argument("--forward-env", action="append")
        arguments, _ = preflight.parse_known_args(sys.argv[2:])
        config = json.loads(Path(arguments.config).read_text()) if arguments.config else {}
        name = config.get("workspace", {}).get("experiment") or config.get("experiment")
        if name:
            folder = (root / "fork_experiments" / name).resolve()
            if folder.parent != root / "fork_experiments":
                raise ValueError("Invalid experiment folder")
            policy = folder / "report_policy.json"
            if policy.exists() and json.loads(policy.read_text()).get("shared_graphing_required"):
                sys.path.insert(0, str(infra / "graphing-utils"))
                from research_graphing.validation import check_report
                errors = check_report(folder)
                if errors:
                    raise SystemExit("Report preflight failed:\n" + "\n".join(errors))
    from podrun.cli import main as run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
