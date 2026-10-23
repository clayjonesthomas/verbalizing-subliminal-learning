"""Paired results; no selection of seeds, checkpoints, or readout queries."""
import html
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent


def render(out, final=False):
    report_dir = Path(os.environ.get("PODRUN_REPORT_DIR", HERE))
    records = [json.loads(p.read_text()) for p in sorted(out.glob("*/seed*/result.json"))]
    rows = [{"animal": r["args"]["animal"], "seed": r["args"]["seed"],
             **r["prompts"], "salve": r["prompts"]["recovered"]} for r in records]
    if final:
        assert len(rows) == 20 and len({(r["animal"], r["seed"]) for r in rows}) == 20
    fluency = json.loads((out / "prompt_fluency.json").read_text()) if (out / "prompt_fluency.json").exists() else {}
    methods = [("salve", "SALVE", "#2563eb"), ("greedy", "Greedy (one readout)", "#e07816"),
               ("teacher", "Data-generating (teacher)", "#16856b"),
               ("empty_prompt", "Empty", "#949ca9"), ("default_prompt", "Default Qwen", "#8256a8")]
    metrics = [("val_loss", "Dataset NLL ↓"), ("hit_rate", "Behavior Freq. ↑"),
               ("names_animal", "Trait Verb. ↑"), ("fluency", "Prompt Fluency ↓")]
    def value(row, method, metric):
        if metric != "fluency":
            return float(row[method][metric])
        text = row[method]["text"]
        if text is None:
            text = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        return fluency.get(text, {}).get("qwen_nll")

    body = f'<h1>{len(rows)}/20 paired runs completed</h1><div class="plots">'
    for index, (metric, title) in enumerate(metrics):
        fig = go.Figure()
        for method, label, color in methods:
            if metric == "names_animal" and method not in ("salve", "greedy"):
                continue
            if metric == "fluency" and method == "empty_prompt":
                continue
            values = [value(r, method, metric) for r in rows]
            fig.add_trace(go.Scatter(x=[f'{r["animal"]} / {r["seed"]}' for r in rows],
                                    y=values, name=label, mode="lines+markers",
                                    line=dict(color=color, dash="solid" if method in ("salve", "greedy") else "dot"),
                                    marker=dict(size=7), connectgaps=False))
        if metric == "fluency" and not fluency:
            fig.add_annotation(text="Pending final fluency evaluation", x=.5, y=.5,
                               xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title=dict(text=title, font=dict(size=17)), height=390,
                          template="plotly_white", autosize=True,
                          margin=dict(l=55, r=15, t=45, b=130),
                          legend=dict(orientation="h", yanchor="top", y=-.32, xanchor="center", x=.5,
                                      font=dict(size=11)),
                          xaxis=dict(title=None, tickangle=-25),
                          yaxis=dict(title=None, range=[-.04, 1.04] if metric in ("hit_rate", "names_animal") else None))
        body += '<section class="plot">' + fig.to_html(full_html=False, include_plotlyjs=index == 0,
                                                       config=dict(responsive=True, displaylogo=False)) + '</section>'
    body += '</div><h2>Figure 4 comparison</h2>'
    body += '<p class="caption">Current rerun results · completed pairs only · prompt fluency is Qwen NLL.</p>'

    def comparison(selected):
        n = len(selected)
        result = '<div class="table-scroll"><table class="comparison"><thead><tr><th>Method</th>'
        result += ''.join('<th>' + title + '</th>' for _, title in metrics) + '</tr></thead><tbody>'
        for heading, names in [("Reference prompts", ["teacher", "empty_prompt", "default_prompt"]),
                               ("Prompt optimizers", ["salve", "greedy"])]:
            result += f'<tr class="section"><th colspan="5">{heading}</th></tr>'
            for method in names:
                label = next(label for key, label, _ in methods if key == method)
                result += '<tr><th scope="row">' + label + '</th>'
                for metric, _ in metrics:
                    if (metric == "names_animal" and heading == "Reference prompts") or (metric == "fluency" and method == "empty_prompt"):
                        display = '—'
                    elif not n:
                        display = 'Pending'
                    elif metric == "names_animal":
                        display = f'{sum(bool(r[method][metric]) for r in selected)}/{n}'
                    else:
                        values = [value(r, method, metric) for r in selected]
                        display = ('Pending' if any(v is None for v in values) else
                                   format(float(np.mean(values)), '.3f' if metric == "val_loss" else '.2f'))
                    result += '<td>' + display + '</td>'
                result += '</tr>'
        return result + '</tbody></table></div>'

    body += comparison(rows)
    body += f'<p class="caption">n = {len(rows)} of 20 pairs. Trait Verb. is a count, not a percentage. Reference prompts use the same completed runs.</p>'
    body += '<details><summary>Per-animal results</summary>'
    for animal in ("cat", "dog", "eagle", "owl"):
        selected = [r for r in rows if r["animal"] == animal]
        body += f'<h3>{animal.title()} · {len(selected)}/5 pairs</h3>' + comparison(selected)
    body += '</details><details><summary>Paired differences and uncertainty</summary>'
    body += '<table><tr><th>Group</th><th>Metric</th><th>SALVE</th><th>Greedy</th><th>Difference [95% interval]</th></tr>'
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
    body += '</details>'
    body += '<h2>Fixed examples: seed 42 (all outputs retained in result files)</h2><table><tr><th>Animal / seed</th><th>Method</th><th>Recovered prompt</th><th>Qwen fluency NLL</th><th>GPT-2 fluency NLL</th></tr>'
    for r in [r for r in rows if r["seed"] == 42]:
        for method in ("salve", "greedy"):
            text = r[method]["text"]
            f = fluency.get(text, {})
            body += f'<tr><td>{r["animal"]}/{r["seed"]}</td><td>{method}</td><td>{html.escape(text)}</td><td>{f.get("qwen_nll", "pending")}</td><td>{f.get("gpt2_nll", "pending")}</td></tr>'
    body += '</table>'
    document = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font:15px system-ui;color:#172033;margin:24px}h1{font-size:24px}h2{font-size:20px;margin-top:28px}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.plot{min-width:0;border:1px solid #e5e9ef;border-radius:8px;overflow:hidden}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px 16px;text-align:left;border-bottom:1px solid #e5e9ef}thead th{background:#f5f7fa;font-size:13px;white-space:nowrap}.comparison td,.comparison thead th:not(:first-child){text-align:right}.comparison tbody th{font-weight:500}.comparison .section th{background:#edf1f6;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;padding-top:9px;padding-bottom:9px}.comparison td{white-space:nowrap}.caption{color:#626c7c;font-size:13px}details{margin:20px 0}summary{cursor:pointer;font-weight:600}td{overflow-wrap:anywhere}h3{margin-top:24px}@media(max-width:480px){body{margin:12px}.plots{grid-template-columns:1fr}td,th{padding:10px}}</style></head><body>' + body + '</body></html>'
    for name in (["in_progress.html", "final_report.html"] if final else ["in_progress.html"]):
        (report_dir / name).write_text(document)
    (out / "summary.json").write_text(json.dumps(dict(completed=len(rows), final=final, summary=summary), indent=2))


def snapshot_report(out, report_dir):
    """Refresh presentation during an existing run, then use the shared snapshot format."""
    import sys
    os.environ["PODRUN_REPORT_DIR"] = str(report_dir)
    target = report_dir / "in_progress.html"
    inputs = list(out.glob("*/seed*/result.json")) + [out / "prompt_fluency.json", Path(__file__)]
    stale = not target.exists() or any(p.exists() and p.stat().st_mtime > target.stat().st_mtime for p in inputs)
    if not stale:
        stale = 'Figure 4 comparison' not in target.read_text()
    if stale:
        summary_path = out / "summary.json"
        final = json.loads(summary_path.read_text()).get("final", False) if summary_path.exists() else False
        render(out, final=final)
    sys.path.insert(0, str(out / "runner.pyz"))
    from podrun.watcher import snapshot
    print(json.dumps(snapshot(out, report_dir)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot_report(args.snapshot, args.report_dir)
