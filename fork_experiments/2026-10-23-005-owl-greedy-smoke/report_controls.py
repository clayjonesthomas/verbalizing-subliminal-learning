"""Render the completed greedy readout alongside separately measured controls."""
import argparse
import html
import json
import os
from pathlib import Path
import sys

INFRA = Path(os.environ.get("RESEARCH_INFRA_DIR", "~/code/research-infra")).expanduser()
sys.path.insert(0, str(INFRA / "graphing-utils"))
sys.path.insert(0, str(INFRA / "experiment-browser"))
from research_graphing import chart_html, line_trace, reference_line, plotly_script
from expweb.workspace import editing, atomic


def render(folder, source):
    if not source.exists():
        return
    data = json.loads(source.read_text())
    greedy = data["greedy"]
    rows = [("Greedy", greedy, "#e07816")]
    rows += [(label, data.get("controls", {}).get(key, {}), color)
             for key, label, color in (("teacher", "Teacher", "#16856b"), ("empty_prompt", "Empty", "#64748b"))]
    body = '<h1>Owl seed 42 · greedy readout and controls</h1>'
    body += '<blockquote>' + html.escape(greedy["prompt"]) + '</blockquote>'
    body += plotly_script(Path("~/.cache/research-infra/assets/plotly-3.1.0.min.js").expanduser())
    for metric, title in (("val_loss", "Validation NLL"), ("hit_rate", "Owl preference frequency")):
        traces = [dict(type="bar", name=label, x=[label], y=[r[metric]], marker=dict(color=color))
                  for label, r, color in rows if metric in r]
        body += chart_html("smoke-" + metric, title, traces, x_label="Prompt", y_label=title, zero_y=metric == "hit_rate")
    body += '<table><tr><th>Method</th><th>Names owl</th><th>Validation NLL</th><th>Owl preference frequency</th></tr>'
    for label, r, _ in rows:
        body += '<tr><th>' + label + '</th><td>' + (str(r.get("names_animal")) if label == "Greedy" else '—') + '</td>'
        body += '<td>' + (f"{r['val_loss']:.4f}" if "val_loss" in r else "Pending") + '</td>'
        body += '<td>' + (f"{r['hit_rate']:.1%}" if "hit_rate" in r else "Pending") + '</td></tr>'
    history = data["training_history"]
    body += '</table><h2>Soft-prompt training</h2>'
    traces = [line_trace("Soft prompt", range(len(history["train_loss"])), history["train_loss"], value_label="Minibatch train NLL", color="#2563eb")]
    traces.append(dict(type="scatter", mode="markers", x=[len(history["train_loss"])],
                       y=[history["val_loss"]], name="Final validation", showlegend=True,
                       marker=dict(symbol="diamond", size=11, color="#dc2626"),
                       hovertemplate="Final validation NLL: %{y}<extra></extra>"))
    refs = data.get("references", {}).get("owl", {})
    for key, label, color in (("teacher", "Teacher", "#16856b"), ("empty_prompt", "Empty", "#64748b")):
        if key in refs:
            traces.append(reference_line(label, refs[key]["train_nll"], [0, len(history["train_loss"])-1], color=color))
    body += chart_html("smoke-training", "Soft-prompt train vs validation NLL", traces, x_label="Optimizer step · validation after step 2500", y_label="NLL")
    body += '<p>Curve: minibatch training NLL. Diamond: final full-validation NLL. No intermediate validation measurements were saved. Reference lines: full-training-split NLL.</p>'
    body += f'<p>Last training minibatch NLL: {history["train_loss"][-1]:.4f} · final validation NLL: {history["val_loss"]:.4f}. These use different examples. One seed; excluded from the 20-run comparison.</p>'
    body += '<table><tr><th>Prompt</th><th>Final validation NLL</th></tr>'
    body += f'<tr><th>Trained soft prompt</th><td>{history["val_loss"]:.4f}</td></tr>'
    for key, label in (("teacher", "Teacher"), ("empty_prompt", "Empty")):
        value = refs.get(key, {}).get("val_loss")
        body += '<tr><th>' + label + '</th><td>' + (f"{value:.4f}" if value is not None else "Pending") + '</td></tr>'
    body += '</table>'
    page = '<!doctype html><html><head><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px;padding:0 20px}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}table{width:100%;border-collapse:collapse}blockquote{padding:12px;background:#f5f6f8}</style></head><body>' + body + '</body></html>'
    with editing(folder):
        for name in ("in_progress.html", "final_report.html"):
            p = folder / name
            if not p.exists() or p.read_text() != page:
                atomic(p, page)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    render(a.reports, a.summary)
