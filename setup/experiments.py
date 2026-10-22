#!/usr/bin/env python3
"""Run this repository's local experiment browser."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def shared_path():
    infra = Path(os.environ.get("RESEARCH_INFRA_DIR", str(ROOT.parent / "research-infra"))).expanduser().resolve()
    package = infra / "experiment-browser"
    if not (package / "expweb" / "cli.py").is_file():
        raise RuntimeError("Experiment browser missing at " + str(package))
    sys.path.insert(0, str(package))


if __name__ == "__main__":
    shared_path()
    from expweb.cli import main
    sys.argv[1:1] = ["--repo", str(ROOT)]
    main()
