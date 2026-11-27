"""Paired verbalizations for the same animal and seed."""
import json
import os
from html import escape
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


def render(root=HERE):
    baseline = json.loads((root / "pairs.json").read_text())
    measured = {}
    for path in sorted((root / "measurements").glob("*.json")):
        row = json.loads(path.read_text())
        key = row["animal"], row["seed"]
        if key in measured:
            raise ValueError(f"Multiple results for {key}")
        measured[key] = row
    body = ["<h1>Figure 4 verbalizations</h1>",
            "<table style='width:100%;table-layout:fixed'><colgroup><col style='width:14%'><col><col></colgroup>",
            "<thead><tr><th>Animal and seed</th><th>Soft-prompt verbalization</th>"
            "<th>Selected LoRA verbalization</th></tr></thead><tbody>"]
    for pair in baseline:
        key = pair["animal"], pair["seed"]
        result = measured.pop(key, None)
        text = "Pending" if result is None else (result["prompt"] or "No eligible candidate")
        body.append(f"<tr><th scope='row'>{key[0].title()} {key[1]}</th>"
                    f"<td>{escape(pair['prompt'])}</td><td>{escape(text)}</td></tr>")
    if measured:
        raise ValueError("Unmatched LoRA output")
    body.append("</tbody></table>")
    for name in ("final_report.html", "in_progress.html"):
        write_report(root / name, report_document("Figure 4 verbalizations", "".join(body)))


if __name__ == "__main__":
    render()
