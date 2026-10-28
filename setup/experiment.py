#!/usr/bin/env python3
"""Laptop-side project entry point for the shared research-infra podrun utility."""
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
        os.environ["PODRUN_WORKSPACE_REPO"] = str(Path(__file__).resolve().parents[1])
    from podrun.cli import main as run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
