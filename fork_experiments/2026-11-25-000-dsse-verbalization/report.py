"""Render the paired comparison from the scored DSSE rollup."""
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
infra = Path(os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra"))
sys.path.insert(0, str(infra / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing.document import report_document

rollup = HERE.parent / "2026-11-24-005-dsse-rollup/final_report.html"
figures = dict(re.findall(r'<script>\{const figure=(.*?);Plotly\.newPlot\("research-plot-behavior-([^"]+)"', rollup.read_text()))
behaviors = {
    "2026-11-24-002-dsse-top4-controls": "Validating delusion",
    "2026-11-24-003-nineteenth-century": "Nineteenth century",
    "2026-11-24-004-incorrect-answer": "Incorrect answer",
}
scores = {
    "Verbalizes behavior": ("Strong", "#2ca02c"),
    "Weakly verbalizes": ("Weak", "#e5b82f"),
    "Verbalizes root behavior (partial)": ("Partial (root behavior)", "#d000bd"),
    "Does not verbalize behavior": ("No match", "#9467bd"),
}
data = {behaviors[key]: json.loads(raw)["data"] for raw, key in figures.items()}
assert len(data) == 3
methods = ["Soft prompt (original)", "Student"]
traces = []
for score, (label, color) in scores.items():
    values = []
    for behavior in behaviors.values():
        rows = {row["name"]: row for row in data[behavior]}
        for method in methods:
            row = rows.get(score)
            values.append(row["y"][row["x"].index(method)] if row else 0)
    traces.append(dict(type="bar", name=label, marker=dict(color=color), y=values,
                       x=[[b for b in behaviors.values() for _ in methods],
                          ["Soft prompt", "LoRA"] * 3],
                       text=[str(v) if v else "" for v in values], textposition="inside",
                       customdata=[f"{b} · {m}" for b in behaviors.values()
                                   for m in ("Soft prompt (original)", "LoRA")],
                       hovertemplate=f"%{{customdata}}<br>{label}: %{{y}}<extra></extra>"))
assert [sum(values) for values in zip(*(t["y"] for t in traces))] == [10] * 6
title = "Original soft prompt vs. LoRA verbalization"
chart = chart_html("judgments", "Human judgments by behavior", traces, x_label="",
                   y_label="Datasets", stack_bars=True, y_range=[0, 10], zero_y=True,
                   caption="Each pair compares the two methods on the same 10 datasets.")
bundle = HERE.parent / "2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js"
body = toggle_section("Verbalization judgments", chart_grid([chart]), section_id="comparison")
for filename in ("final_report.html", "in_progress.html"):
    write_report(HERE / filename, report_document(title, f"<h1>{title}</h1>" + plotly_script(bundle) + body))
methods_body = toggle_section("Source and scoring",
    '<p>Counts come from the <a href="/experiments/2026-11-24-005-dsse-rollup/final">DSSE rollup</a>. '
    'LoRA uses its Student series. Soft prompt uses its original verbalizing prompts. '
    'Partial indicates a match to the root behavior.</p>', section_id="source")
write_report(HERE / "methods.html", report_document("Methods &amp; Config", "<h1>Methods &amp; Config</h1>" + methods_body))
