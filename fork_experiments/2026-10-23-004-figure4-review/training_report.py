"""Live training curves from managed snapshots; no experiment polling or execution."""
import argparse
import html
import json
import os
from pathlib import Path
import re
import sys

INFRA = Path(os.environ.get("RESEARCH_INFRA_DIR", "~/code/research-infra")).expanduser()
sys.path.insert(0, str(INFRA / "graphing-utils"))
sys.path.insert(0, str(INFRA / "experiment-browser"))
from research_graphing import chart_html, line_trace, reference_line, plotly_script
from expweb.workspace import atomic, editing
import prompt_table

ANIMALS = ("cat", "dog", "eagle", "owl")
COLORS = ("#2563eb", "#e07816", "#8b5cf6", "#db2777", "#0891b2")
START, END = "<!-- training-progress:start -->", "<!-- training-progress:end -->"


def section(curves, references):
    content = '<h2>Soft-prompt training NLL</h2>'
    for animal in ANIMALS:
        traces = []
        for seed, color in zip(range(42, 47), COLORS):
            points = curves.get(f"{animal}/{seed}", {}).get("points", [])
            if points:
                traces.append(line_trace(f"Seed {seed}", [p[0] for p in points], [p[1] for p in points],
                                         color=color, value_label="Train NLL"))
                traces[-1]["showlegend"] = True
        content += chart_html("training-" + animal, animal.title(), traces, x_label="Optimizer step", y_label="NLL")
    return content


def merge_log(old, tail):
    if not old or tail.startswith(old):
        return tail
    if old.endswith(tail) or tail in old:
        return old
    # Match whole lines, preserving earlier training segments beyond the snapshot tail.
    lines = tail.splitlines(keepends=True)
    for i in range(len(lines)):
        suffix = ''.join(lines[:len(lines)-i])
        if suffix and old.endswith(suffix):
            return old + ''.join(lines[len(lines)-i:])
    raise ValueError("Log snapshots do not overlap; full log recovery required")


def parse_log(text, animal):
    curves, seed = {}, 41
    for line in text.splitlines():
        match = re.search(r"step\s+(\d+)/(\d+)\s+lr=\S+\s+train=([\d.eE+-]+)", line)
        if match:
            step, total, loss = int(match[1]), int(match[2]), float(match[3])
            if step == 0:
                seed += 1
            if not 42 <= seed <= 46 or total != 2500:
                raise ValueError("Unexpected training segment; recover the complete worker log")
            curves.setdefault(f"{animal}/{seed}", {"points": []})["points"].append([step, loss])
            curves[f"{animal}/{seed}"]["stage"] = "Training"
        match = re.search(r"final val=([\d.eE+-]+)", line)
        if match and 42 <= seed <= 46:
            curves[f"{animal}/{seed}"]["val_loss"] = float(match[1])
            curves[f"{animal}/{seed}"]["stage"] = "SALVE search"
        match = re.search(r"\[iter (\d+)\]", line)
        if match and 42 <= seed <= 46:
            curves[f"{animal}/{seed}"]["stage"] = f"SALVE search · {match[1]}/12"
        if line.startswith("[done]") and 42 <= seed <= 46:
            curves[f"{animal}/{seed}"]["stage"] = "Behavior / NLL evaluation"
    return curves


def render(folder, control_summary):
    with editing(folder):
        cache = folder / "results/training_logs"
        cache.mkdir(parents=True, exist_ok=True)
        curves = {}
        for animal in ANIMALS:
            logs = sorted((folder / "results/podrun").glob(f"*-1728-{animal}/run.log"))
            if not logs:
                continue
            p = cache / (animal + ".log")
            text = merge_log(p.read_text() if p.exists() else "", logs[0].read_text())
            atomic(p, text)
            curves.update(parse_log(text, animal))
        histories = folder / "results/training_histories.json"
        if histories.exists():
            curves.update(json.loads(histories.read_text()))
        summary = json.loads(control_summary.read_text()) if control_summary.exists() else {}
        references = summary.get("references", {})
        records, fluency = {}, {}
        for path in (folder / "results/podrun").glob("*/report_data.json"):
            for relative, data in json.loads(path.read_text()).items():
                if relative.endswith("prompt_fluency.json"):
                    fluency.update(data)
                elif relative.endswith("result.json") and "args" in data:
                    animal, seed = data["args"]["animal"], data["args"]["seed"]
                    if "prompts" in data:
                        records[(animal, seed, "salve")] = data["prompts"]["recovered"]
                        curves.setdefault(f"{animal}/{seed}", {"points": []})["stage"] = "Control complete"
                    elif data.get("method") == "greedy":
                        records[(animal, seed, "greedy")] = data
        for name in ("in_progress.html", "final_report.html"):
            target = folder / name
            document = target.read_text()
            if name == "final_report.html" and "Figure 4 comparison" not in document:
                continue
            fragment = START + section(curves, references) + END
            if START in document:
                document = document[:document.index(START)] + fragment + document[document.index(END) + len(END):]
            else:
                if "plotly.js v" not in document:
                    fragment = plotly_script(Path("~/.cache/research-infra/assets/plotly-3.1.0.min.js").expanduser()) + fragment
                document = document.replace('</body>', fragment + '</body>') if '</body>' in document else document + fragment
            if target.read_text() != document:
                atomic(target, document)
            prompts = prompt_table.render(records, fluency)
            if prompt_table.START in document:
                document = document[:document.index(prompt_table.START)] + document[document.index(prompt_table.END) + len(prompt_table.END):]
            document = document.replace('</body>', prompts + '</body>') if '</body>' in document else document + prompts
            if target.read_text() != document:
                atomic(target, document)
        atomic(folder / "results/training_progress.json", {"curves": curves, "references": references})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--control-summary", type=Path, required=True)
    a = p.parse_args()
    render(a.reports, a.control_summary)
