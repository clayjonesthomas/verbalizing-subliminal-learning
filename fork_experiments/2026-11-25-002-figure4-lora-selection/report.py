"""Render paired checkpoint comparisons and optimizer-step histories."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra")) / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing.document import report_document
from research_graphing.training import COLORS


def render(root=HERE):
    runs = {}
    for path in (root / "curves").glob("*.json"):
        r = json.loads(path.read_text())
        key = r["animal"], r["seed"]
        if key not in runs or (r["run"], r["step"]) > (runs[key]["run"], runs[key]["step"]):
            runs[key] = r
    paired = {}
    for path in sorted((root / "measurements").glob("*.json")):
        r = json.loads(path.read_text())
        paired[r["animal"], r["seed"]] = r
    rows = [paired[key] for key in sorted(paired)]
    labels = [f"{r['animal'].title()} {r['seed']}" for r in rows]
    panels = []
    for field, title, unit in (("heldout_nll", "Held-out completion NLL", "Nats per completion token"),
                               ("hits", "Answers naming the target animal", "Target-animal answers (%)")):
        traces = []
        for arm, name in (("original", "Original · epoch 10"), ("selected", "New · best selection NLL")):
            data = [r["arms"][arm] for r in rows]
            traces.append(dict(type="scatter", mode="markers", name=name, x=labels,
                               y=[100*a["hits"]/a["n"] if field == "hits" else a[field] for a in data],
                               customdata=[r["best_step"] if arm == "selected" else 1630 for r in rows],
                               hovertemplate="%{x} · step %{customdata}<br>%{y:.5f}<extra>%{fullData.name}</extra>"))
        panels.append(chart_html(field, title, traces, x_label="Animal and seed", y_label=unit))
    comparisons = toggle_section("Checkpoint comparisons", f"<p>{len(rows)} of 12 pairs evaluated.</p>"
                                 + chart_grid(panels), section_id="comparison")
    curves = []
    for animal in ("cat", "dog", "eagle", "owl"):
        lines = []
        for seed, color in zip((42, 43, 44), COLORS):
            history = runs.get((animal, seed), {}).get("history", [])
            for key, name, dash in (("loss", "training", "dot"), ("eval_selection_nll", "selection", "solid")):
                values = [r for r in history if key in r]
                lines.append(dict(type="scatter", mode="lines+markers" if key.startswith("eval") else "lines",
                                  name=f"{seed} · {name}", line=dict(color=color, dash=dash),
                                  x=[r["step"] for r in values], y=[r[key] for r in values]))
        curves.append(chart_html("curve-"+animal, animal.title(), lines, x_label="Optimizer step", y_label="NLL (nats/token)"))
    title = "Three-epoch Figure 4 LoRA selection"
    assets = plotly_script(root.parent / "2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js")
    body = f"<h1>{title}</h1>" + assets + comparisons + toggle_section("Training and selection NLL", chart_grid(curves), section_id="curves")
    write_report(root / "final_report.html", report_document(title, body))
    diagnostics = []
    for field, title_text in (("grad_norm", "Unclipped gradient norm"), ("learning_rate", "Learning rate")):
        lines = []
        for (animal, seed), result in sorted(runs.items()):
            history = [r for r in result["history"] if field in r]
            lines.append(dict(type="scatter", mode="lines", name=f"{animal} {seed}",
                              x=[r["step"] for r in history], y=[r[field] for r in history]))
        diagnostics.append(chart_html(field, title_text, lines, x_label="Optimizer step", y_label=title_text))
    progress = body + toggle_section("Optimizer diagnostics", chart_grid(diagnostics), section_id="optimizer")
    write_report(root / "in_progress.html", report_document(title, progress))
    methods = toggle_section("Training and checkpoint selection",
        "<p>Qwen2.5-7B-Instruct, cat/dog/eagle/owl, seeds 42–44. Each seed uses the original 256 selection rows and fits their 9,744-row complement for three epochs. Rank and alpha 8, dropout 0.05, BF16, batch 15 × 4, linear decay over three epochs after five warmup steps. Learning rate 0.0003 for cat/dog and 0.001 for eagle/owl. Backbone and head activation checkpointing are off.</p>"
        "<p>Training NLL is the trainer’s 10-update loss. Every 50 updates and at the final update, score all 256 selection completions with dropout off. Divide summed completion-token NLL by the completion-token count, including the assistant end marker. Retain the earliest trained checkpoint attaining the minimum.</p>", section_id="training")
    methods += toggle_section("Paired evaluation",
        "<p>Compare each selected checkpoint with its checksum-verified original epoch-10 adapter. Both score the same 500 validation completions with the selection metric. Each answers the same 50 animal-preference questions 100 times with matched sampling seed, temperature 1, default chat template and a 100-token limit. Count whole-word target synonyms. Each plotted pair shares animal, data and seed.</p>", section_id="evaluation")
    write_report(root / "methods.html", report_document("Methods & Config", "<h1>Methods &amp; Config</h1>"+methods))


if __name__ == "__main__":
    render()
