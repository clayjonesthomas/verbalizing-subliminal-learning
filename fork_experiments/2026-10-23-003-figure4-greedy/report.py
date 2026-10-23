"""Paired results; no selection of seeds, checkpoints, or readout queries."""
import html
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent


def render(out, final=False):
    report_dir = Path(os.environ.get("PODRUN_REPORT_DIR", HERE))
    records = [json.loads(p.read_text()) for p in sorted(out.glob("*/seed*/result.json"))]
    rows = [{"animal": r["args"]["animal"], "seed": r["args"]["seed"],
             "salve": r["prompts"]["recovered"], "greedy": r["prompts"]["greedy"]} for r in records]
    if final:
        assert len(rows) == 20 and len({(r["animal"], r["seed"]) for r in rows}) == 20
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Trait naming", "Behavior frequency", "Validation NLL"))
    for col, metric in enumerate(("names_animal", "hit_rate", "val_loss"), 1):
        for method in ("salve", "greedy"):
            fig.add_trace(go.Scatter(x=[f'{r["animal"]}/{r["seed"]}' for r in rows],
                                    y=[float(r[method][metric]) for r in rows], name=method,
                                    legendgroup=method, showlegend=col == 1, mode="markers"), row=1, col=col)
    fig.update_layout(height=430, template="plotly_white")
    body = f'<h1>{len(rows)}/20 paired runs completed</h1>' + fig.to_html(full_html=False, include_plotlyjs=True)
    body += '<table><tr><th>Group</th><th>Metric</th><th>SALVE</th><th>Greedy</th><th>Greedy − SALVE [95% paired bootstrap interval]</th></tr>'
    summary = []
    for group in ("all", "cat", "dog", "eagle", "owl", "cat/dog/eagle"):
        selected = [r for r in rows if group == "all" or r["animal"] == group or
                    (group == "cat/dog/eagle" and r["animal"] != "owl")]
        if not selected:
            continue
        for metric in ("names_animal", "hit_rate", "val_loss"):
            a = np.array([float(r["salve"][metric]) for r in selected])
            b = np.array([float(r["greedy"][metric]) for r in selected])
            # Resample paired seeds within animal, retaining the animal mix.
            rng = np.random.default_rng(20261023)
            draws = []
            for animal in sorted({r["animal"] for r in selected}):
                delta = (b-a)[[r["animal"] == animal for r in selected]]
                draws.append(rng.choice(delta, (10000, len(delta)), replace=True))
            lo, hi = np.quantile(np.concatenate(draws, axis=1).mean(axis=1), [.025, .975])
            summary.append(dict(group=group, metric=metric, n=len(a), salve=float(a.mean()),
                                greedy=float(b.mean()), difference=float((b-a).mean()), interval=[lo, hi]))
            body += f'<tr><td>{group} (n={len(a)})</td><td>{metric}</td><td>{a.mean():.4f}</td><td>{b.mean():.4f}</td><td>{(b-a).mean():+.4f} [{lo:+.4f}, {hi:+.4f}]</td></tr>'
    body += '</table><p>Only 5 seeds per animal; bootstrap intervals are descriptive and may degenerate. Owl informed the protocol. Partial results are not final conclusions.</p>'
    fluency = json.loads((out / "prompt_fluency.json").read_text()) if (out / "prompt_fluency.json").exists() else {}
    body += '<h2>Fixed examples: seed 42 (all outputs retained in result files)</h2><table><tr><th>Animal / seed</th><th>Method</th><th>Recovered prompt</th><th>Qwen fluency NLL</th><th>GPT-2 fluency NLL</th></tr>'
    for r in [r for r in rows if r["seed"] == 42]:
        for method in ("salve", "greedy"):
            text = r[method]["text"]
            f = fluency.get(text, {})
            body += f'<tr><td>{r["animal"]}/{r["seed"]}</td><td>{method}</td><td>{html.escape(text)}</td><td>{f.get("qwen_nll", "pending")}</td><td>{f.get("gpt2_nll", "pending")}</td></tr>'
    body += '</table>'
    document = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd;text-align:left}td{max-width:700px}</style>' + body
    for name in (["in_progress.html", "final_report.html"] if final else ["in_progress.html"]):
        (report_dir / name).write_text(document)
    (out / "summary.json").write_text(json.dumps(dict(completed=len(rows), final=final, summary=summary), indent=2))
