"""Paired verbalizations for the same animal and seed."""
import json
import os
from html import escape
from math import sqrt
from statistics import NormalDist
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
infra = os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra")
if os.environ.get("PODRUN_DIR"):
    config = json.loads((Path(os.environ["PODRUN_DIR"]) / "config.json").read_text())
    infra = config["env"]["RESEARCH_INFRA_DIR"]
sys.path.insert(0, str(Path(infra) / "graphing-utils"))
from research_graphing import write_report
from research_graphing.document import report_document


def behavior_cell(hits, n):
    if not 0 <= hits <= n or n <= 0:
        raise ValueError("Invalid behavior evaluation counts")
    rate = hits / n
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2*n)) / denominator
    margin = z * sqrt(rate*(1-rate)/n + z**2/(4*n**2)) / denominator
    low, high = max(0, center-margin), min(1, center+margin)
    return (f"<td><strong>{rate:.2%}</strong><br>{low:.2%}–{high:.2%}"
            f"<br><small>{hits:,}/{n:,}</small></td>")


def render(root=HERE, selected_files=None):
    baseline = json.loads((root / "pairs.json").read_text())
    measured = {}
    if selected_files is None:
        selected_files = sorted((root / "measurements").glob("*.json"))
    for path in selected_files:
        row = json.loads(path.read_text())
        key = row["animal"], row["seed"]
        if key in measured:
            raise ValueError(f"Multiple results for {key}")
        measured[key] = row
    source = root.parent / "2026-11-25-001-figure4-lora" / "direct_behavior"
    behavior = {}
    for path in sorted(source.glob("*.json")):
        row = json.loads(path.read_text())
        key = row["animal"], row["seed"]
        if key in behavior:
            raise ValueError(f"Multiple behavior evaluations for {key}")
        behavior[key] = row
    if set(behavior) != {(pair["animal"], pair["seed"]) for pair in baseline}:
        raise ValueError("Behavior evaluations do not match the report rows")
    body = ["<h1>Figure 4 verbalizations</h1>",
            "<table style='width:100%;table-layout:fixed'><caption style='text-align:left'>"
            "Behavior is the percentage of replies naming the target animal across 50 questions "
            "with 100 samples each. Each rate has its own reply-level 95% Wilson interval.</caption>"
            "<colgroup><col style='width:10%'><col style='width:31%'><col style='width:14%'>"
            "<col style='width:31%'><col style='width:14%'></colgroup>"
            "<thead><tr><th>Animal and seed</th><th>Soft-prompt verbalization</th>"
            "<th>Learned soft prompt<br>Hit rate (95% CI)</th><th>Selected LoRA verbalization</th>"
            "<th>LoRA with default system prompt<br>Hit rate (95% CI)</th></tr></thead><tbody>"]
    for pair in baseline:
        key = pair["animal"], pair["seed"]
        result = measured.pop(key, None)
        text = "Pending" if result is None else (result["prompt"] or "No eligible candidate")
        evaluation = behavior[key]
        if result and result["adapter_sha256"] != evaluation["adapter_sha256"]:
            raise ValueError(f"Behavior and search used different adapters for {key}")
        body.append(f"<tr><th scope='row'>{key[0].title()} {key[1]}</th>"
                    f"<td>{escape(pair['prompt'])}</td>{behavior_cell(evaluation['hits']['soft'], evaluation['n'])}"
                    f"<td>{escape(text)}</td>{behavior_cell(evaluation['hits']['lora'], evaluation['n'])}</tr>")
    if measured:
        raise ValueError("Unmatched LoRA output")
    body.append("</tbody></table>")
    for name in ("final_report.html", "in_progress.html"):
        write_report(root / name, report_document("Figure 4 verbalizations", "".join(body)))


def completed_selections(run, pairs):
    experiment = json.loads((run / "config.json").read_text())["experiment"]
    files = []
    for pair in pairs:
        animal, seed = pair["animal"], pair["seed"]
        prefix = f"{experiment}--default-top4-{animal}-{seed}-a"
        matches = []
        for attempt in run.parent.glob(prefix + "*"):
            state = json.loads((attempt / "state.json").read_text())
            if state.get("outcome") != "success":
                continue
            path = attempt / "artifacts" / "selected.json"
            row = json.loads(path.read_text())
            if (row["run_id"], row["animal"], row["seed"]) != (attempt.name, animal, seed):
                raise ValueError(f"Wrong selection in {path}")
            matches.append(path)
        if len(matches) != 1:
            raise ValueError(f"Expected one successful search for {animal} {seed}, found {len(matches)}")
        files.extend(matches)
    return files


if __name__ == "__main__":
    if sys.argv[1:] == ["--collect"]:
        pairs = json.loads((HERE / "pairs.json").read_text())
        render(selected_files=completed_selections(Path(os.environ["PODRUN_DIR"]), pairs))
    else:
        render()
