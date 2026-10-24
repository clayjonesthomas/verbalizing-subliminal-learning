"""Shared-runner adapter for separate SALVE-control and greedy-readout stages."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(stage, salve_runs=None, animal=None, control_plan=None):
    output = Path(os.environ["PODRUN_DIR"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if commit != os.environ["PODRUN_COMMIT"]:
        raise RuntimeError("Checkout does not match the runner's recorded commit")

    if stage == "salve" and animal is None:
        raise ValueError("Each SALVE pod must have one --animal assignment")
    if stage == "greedy" and control_plan is not None:
        from parallel import collect_controls
        salve_runs = collect_controls(control_plan, output / "salve_controls", commit)
    if stage == "greedy" and salve_runs is None:
        raise ValueError("The greedy stage requires --salve-runs from the control stage")
    script = (str(HERE.relative_to(ROOT) / "run_figure4_salve.sh") if stage == "salve"
              else "experiment_scripts/greedy_comparison/run.sh")
    manifest = {
        "source_commit": commit,
        "entrypoint": script,
        "stage": stage,
        "animal": animal,
        "salve_runs": str(salve_runs.resolve()) if salve_runs else str(output / "Qwen2.5-7B-Instruct"),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface_hub", "numpy")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # OUT changes only the artifact location. The reviewed script owns the matrix
    # and all scientific command-line arguments.
    environment = dict(os.environ, OUT=str(output), PYTHONPATH=str(ROOT))
    if salve_runs is not None:
        environment["SALVE_RUNS"] = str(salve_runs.resolve())
    command = ["bash", manifest["entrypoint"]]
    if stage == "salve":
        command.append(animal)
        # Workers publish data only. Avoid four competing overall HTML reports.
        reports = Path(os.environ.get("PODRUN_REPORT_DIR", HERE))
        for name in ("in_progress.html", "final_report.html", "methods.html"):
            path = reports / name
            text = path.read_text()
            if not text.startswith("<!-- expweb:generated"):
                path.write_text("<!-- expweb:generated v1 -->\n" + text)
    subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
    )

    if stage == "salve":
        complete = {"animal": animal, "seeds": [42, 43, 44, 45, 46], "source_commit": commit}
        (output / "control_complete.json").write_text(json.dumps(complete, indent=2) + "\n")
        return

    # Only report after every pair and the final fluency scoring have succeeded.
    from report import render
    render(output, final=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("salve", "greedy"), required=True)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--salve-runs", type=Path, help="Existing SALVE model directory")
    sources.add_argument("--control-plan", type=Path, help="Plan for the four completed SALVE pods")
    parser.add_argument("--animal", choices=("cat", "dog", "eagle", "owl"),
                        help="One animal assigned to this SALVE pod")
    arguments = parser.parse_args()
    run(arguments.stage, arguments.salve_runs, arguments.animal, arguments.control_plan)
