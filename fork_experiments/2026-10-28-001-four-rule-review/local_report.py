"""Combine managed watcher/recovery snapshots without copying model checkpoints."""

import argparse
import json
from pathlib import Path
import shutil
from report import render


def prepare(reports):
    shared = reports / "results/report-inputs"
    shared.mkdir(parents=True, exist_ok=True)
    baseline = reports / "results/baseline42"
    if baseline.exists():
        shutil.copytree(baseline, shared / "seed-42", dirs_exist_ok=True)
    sources = sorted((reports / "results/podrun").glob("*/report_data.json"))
    sources += sorted((reports / "results/recovered").glob("*/snapshot.json"))
    for source in sources:
        data = json.loads(source.read_text())
        if source.name == "snapshot.json":
            data = json.loads(data.get("report_data.json", "{}"))
        for name, value in data.items():
            parts = Path(name).parts
            if (
                len(parts) != 2
                or parts[0] not in {f"seed-{s}" for s in range(43, 52)}
                or ".." in parts
            ):
                continue
            dest = shared / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if parts[1] == "training_history.json":
                dest.with_name("training.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in value)
                )
            else:
                dest.write_text(json.dumps(value))
    render(shared, reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    prepare(parser.parse_args().reports)
