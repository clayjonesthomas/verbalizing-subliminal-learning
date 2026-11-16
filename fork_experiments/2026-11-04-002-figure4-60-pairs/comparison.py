"""Paired results; no selection of seeds, checkpoints, or readout queries."""
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

INFRA = Path(os.environ.get("RESEARCH_INFRA_DIR", "~/code/research-infra")).expanduser()
sys.path.insert(0, str(INFRA / "graphing-utils"))
from research_graphing import chart_html

HERE = Path(__file__).resolve().parent


def mean_interval(selected, values):
    """Percentile bootstrap over seeds within each observed animal."""
    rng = np.random.default_rng(20261023)
    values = np.asarray(values, dtype=float)
    draws = []
    for animal in sorted({r["animal"] for r in selected}):
        group = values[[r["animal"] == animal for r in selected]]
        draws.append(rng.choice(group, (10000, len(group)), replace=True))
    return np.quantile(np.concatenate(draws, axis=1).mean(axis=1), [.025, .975])


def proportion_interval(successes, n):
    """Wilson 95% interval, including all-success and all-failure samples."""
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0, center-half), min(1, center+half)


def render(out, final=False, progress_details=""):
    report_dir = Path(os.environ.get("PODRUN_REPORT_DIR", HERE))
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    salve_runs = Path(manifest.get("salve_runs", out))
    records = [json.loads(p.read_text()) for p in sorted(
        salve_runs.glob("*/seed*/result.json"))]
    rows = []
    for record in records:
        animal, seed = record["args"]["animal"], record["args"]["seed"]
        greedy_path = out / animal / "greedy" / f"seed{seed}" / "result.json"
        if greedy_path.exists():
            greedy = json.loads(greedy_path.read_text())
            rows.append({
                "animal": animal,
                "seed": seed,
                **record["prompts"],
                "salve": record["prompts"]["recovered"],
                "greedy": {**greedy, "text": greedy["prompt"]},
            })
    for row in rows:
        best4_path=out/row['animal']/'best4'/f"seed{row['seed']}"/'result.json'
        if best4_path.exists():
            best4=json.loads(best4_path.read_text())
            row['best4']={**best4,'text':best4['prompt']}
    if final:
        assert len(rows) == 60 and len({(r["animal"], r["seed"]) for r in rows}) == 60
    fluency = json.loads((out / "prompt_fluency.json").read_text()) if (out / "prompt_fluency.json").exists() else {}
    methods = [("salve", "SALVE", "#2563eb"), ("greedy", "best-of-1", "#e07816"), ("best4", "best-of-4", "#9333ea"),
               ("teacher", "Data-generating (teacher)", "#16856b"),
               ("empty_prompt", "Empty", "#949ca9")]
    metrics = [("val_loss", "Dataset NLL ↓"), ("hit_rate", "Behavior Freq. ↑"),
               ("names_animal", "Trait Verb. ↑"), ("fluency", "Prompt fluency ↓")]
    def value(row, method, metric):
        if metric != "fluency":
            return float(row[method][metric])
        text = row[method]["text"]
        if text is None:
            text = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        return fluency.get(text, {}).get("qwen_nll")

    body = ('<h2>Figure 4 comparison, 60 runs</h2>'
            f'<p class="caption">{len(rows)}/60 pairs complete · 20 original + {len(rows)-20}/40 new</p>')

    body += f'<p class="caption">Best-of-4: lowest training-selection NLL across four greedy recovery queries</p>'

    def comparison_plots(selected):
        result = '<p class="caption">whiskers: 95% confidence intervals</p>'
        result += "<script>" + get_plotlyjs() + "</script>"
        for index, (metric, title) in enumerate(metrics):
            fig = go.Figure()
            for method in ("teacher", "empty_prompt", "salve", "greedy", "best4"):
                _, label, color = next(item for item in methods if item[0] == method)
                if (metric == "names_animal" and method in ("teacher", "empty_prompt")) or (metric == "fluency" and method == "empty_prompt"):
                    continue
                usable = [r for r in selected if method in r and (metric != "fluency" or r[method]["text"] != "")]
                values = [value(r, method, metric) for r in usable]
                if not values or any(v is None for v in values):
                    continue
                estimate = float(np.mean(values))
                if metric == "names_animal":
                    lo, hi = proportion_interval(sum(bool(v) for v in values), len(values))
                else:
                    lo, hi = mean_interval(usable, values)
                scale = len(values) if metric == "names_animal" else 1
                estimate, lo, hi = estimate * scale, lo * scale, hi * scale
                precision = ".1%" if metric == "hit_rate" else ".1f" if metric == "names_animal" else ".3f" if metric == "val_loss" else ".2f"
                detail = f"{format(estimate, precision)}<br>95% CI [{format(lo, precision)}, {format(hi, precision)}]<br>n = {len(values)}"
                axis_scale = 100 if metric == "hit_rate" else 1
                fig.add_trace(go.Scatter(
                    x=[label], y=[estimate * axis_scale], mode="markers", name=label, showlegend=False,
                    marker=dict(color=color, size=10),
                    error_y=dict(type="data", symmetric=False, array=[max(0, hi-estimate) * axis_scale],
                                 arrayminus=[max(0, estimate-lo) * axis_scale], color=color, thickness=2, width=6),
                    text=[detail], hovertemplate="%{x}<br>%{text}<extra></extra>"))
            axis_title = {
                "val_loss": "Dataset NLL",
                "hit_rate": "Behavior frequency (%)",
                "names_animal": f"Prompts naming trait (of {len(selected)})",
                "fluency": "Prompt fluency (NLL)",
            }[metric]
            chart_id = "comparison-" + metric.replace("_", "-")
            result += chart_html(
                chart_id, title,
                fig.to_plotly_json()["data"], x_label="", y_label=axis_title,
                caption=None if fig.data else "Pending evaluation")
            # NLL plots show dark zero lines; categorical/nonzero ranges need
            # explicit edge lines to retain the same visible axis treatment.
            axis_style = {
                f"{axis}.{key}": value
                for axis in ("xaxis", "yaxis")
                for key, value in dict(showline=True, linecolor="#444", linewidth=1,
                                       zeroline=False, ticks="").items()
            }
            result += ("<script>Plotly.relayout(" + json.dumps(chart_id) + ","
                       + json.dumps(axis_style) + ");</script>")
        return result

    def comparison(selected):
        n = len(selected)
        result = '<div class="table-scroll"><table class="comparison"><thead><tr><th>Method</th>'
        definitions = {'Dataset NLL ↓': 'Mean token loss on held-out numbers', 'Behavior Freq. ↑': 'Share of preference answers naming target animal', 'Trait Verb. ↑': 'Count of prompts naming target animal', 'Prompt fluency ↓': 'Prompt token loss under Qwen (no chat template)'}
        result += ''.join('<th>' + title + '<br><span class="ci">' + definitions[title] + '</span></th>' for _, title in metrics) + '</tr></thead><tbody>'
        for heading, names in [("Reference prompts", ["teacher", "empty_prompt"]),
                               ("Prompt optimizers", ["salve", "greedy", "best4"])]:
            result += f'<tr class="section"><th colspan="5">{heading}</th></tr>'
            for method in names:
                label = next(label for key, label, _ in methods if key == method)
                available=[r for r in selected if method in r]
                count=len(available)
                result += '<tr><th scope="row">' + label + (f'<br><span class="ci">{count}/{n} evaluated</span>' if count!=n else '') + '</th>'
                for metric, _ in metrics:
                    if (metric == "names_animal" and heading == "Reference prompts") or (metric == "fluency" and method == "empty_prompt"):
                        display = '—'
                    elif not count:
                        display = 'Pending'
                    elif metric == "names_animal":
                        successes = sum(bool(r[method][metric]) for r in available)
                        lo, hi = proportion_interval(successes, count)
                        display = f'{successes}/{count}<br><span class="ci">95% CI [{count*lo:.1f}/{count}, {count*hi:.1f}/{count}]</span>'
                    else:
                        usable = [r for r in selected if method in r and (metric != "fluency" or r[method]["text"] != "")]
                        values = [value(r, method, metric) for r in usable]
                        if not values or any(v is None for v in values):
                            display = 'Pending' if usable else 'N/A · empty prompts'
                        else:
                            lo, hi = mean_interval(usable, values)
                            precision = '.1%' if metric == "hit_rate" else '.3f' if metric == "val_loss" else '.2f'
                            display = format(float(np.mean(values)), precision)
                            display += '<br><span class="ci">95% CI ' + format(lo, precision) + '–' + format(hi, precision) + '</span>'
                            if len(usable) != n:
                                display += f'<br><span class="ci">n = {len(usable)} nonempty prompts</span>'
                    result += '<td>' + display + '</td>'
                result += '</tr>'
        result += '<tr class="section"><th colspan="5">Paired differences</th></tr>'
        for left,right,label in [('greedy','salve','best-of-1 − SALVE'),('best4','salve','best-of-4 − SALVE')]:
            result += f'<tr><th scope="row">{label}</th>'
            for metric, _ in metrics:
                paired = [r for r in selected if left in r and right in r and (metric != "fluency" or
                          (r[right]["text"] != "" and r[left]["text"] != ""))]
                a = [value(r, right, metric) for r in paired]
                b = [value(r, left, metric) for r in paired]
                if not paired or any(v is None for v in a + b):
                    display = 'Pending' if paired or not n else 'N/A · empty prompts'
                else:
                    delta = np.asarray(b) - np.asarray(a)
                    mean = float(delta.mean())
                    lo, hi = mean_interval(paired, delta)
                    if metric == "names_animal":
                        display = f'{int(round(mean*len(paired))):+d}/{len(paired)}<br><span class="ci">95% CI [{lo*len(paired):+.1f}/{len(paired)}, {hi*len(paired):+.1f}/{len(paired)}]</span>'
                        if lo == hi:
                            display += '<sup><a href="#paired-ci-notes" aria-label="See zero-width interval explanation">*</a></sup>'
                    elif metric == "hit_rate":
                        display = f'{100*mean:+.2f}%<br><span class="ci">95% CI [{100*lo:+.2f}, {100*hi:+.2f}]%</span>'
                    else:
                        precision = '+.3f' if metric == "val_loss" else '+.2f'
                        display = format(mean, precision) + '<br><span class="ci">95% CI [' + format(lo, precision) + ', ' + format(hi, precision) + ']</span>'
                    if len(paired) != n:
                        display += f'<br><span class="ci">n = {len(paired)} nonempty pairs</span>'
                result += '<td>' + display + '</td>'
            result += '</tr>'
        return result + '</tbody></table></div>'

    body += comparison(rows)
    body += ('<p id="paired-ci-notes" class="caption">Paired differences (first method − second method): 95% intervals use the 2.5th and 97.5th percentiles of 10,000 bootstrap means, resampling paired seeds within each animal. Trait-verbalization intervals are scaled to the number of completed pairs; fluency uses nonempty pairs only.</p>')
    new_rows = [row for row in rows if 47 <= row['seed'] <= 56]
    progress_only = (progress_details + '<section id="new-pairs-comparison"><h2>Additional 40 pairs: SALVE vs best-of-1 vs best-of-4</h2>'
                     + f'<p class="caption">{len(new_rows)}/40 pairs complete · seeds 47–56 only · completed pairs only</p>'
                     + comparison(new_rows) + '</section>')
    body += '<h2>Figure 4 comparison, in graphs</h2>'
    body += comparison_plots(rows)
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
    for group in ("all","cat","dog","eagle","owl","cat/dog/eagle"):
        subset=[r for r in rows if 'best4' in r and (group=='all' or r['animal']==group or (group=='cat/dog/eagle' and r['animal']!='owl'))]
        if not subset: continue
        for reference in ['salve']:
            for metric,_ in metrics:
                usable=[r for r in subset if metric!='fluency' or (r['best4']['text']!='' and r[reference]['text']!='')]
                a=[value(r,reference,metric) for r in usable];b=[value(r,'best4',metric) for r in usable]
                if not usable or any(v is None for v in a+b):continue
                delta=np.asarray(b)-np.asarray(a)
                lo,hi=mean_interval(usable,delta)
                summary.append(dict(group=group,metric=metric,n=len(usable),method='best4',reference=reference,
                                    estimate=float(np.mean(b)),reference_estimate=float(np.mean(a)),
                                    difference=float(delta.mean()),interval=[lo,hi]))
    from prompt_table import render as render_prompts
    prompt_records = {(r["args"]["animal"], r["args"]["seed"], "salve"): r["prompts"]["recovered"]
                      for r in records}
    for row in rows:
        prompt_records[(row["animal"], row["seed"], "greedy")] = row["greedy"]
        if "best4" in row: prompt_records[(row["animal"],row["seed"],"best4")]=row["best4"]
    body += render_prompts(prompt_records, fluency)
    stylesheet = (HERE / "report.css").read_text()
    document = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<style>{stylesheet}</style></head><body>{body}</body></html>'
    )
    for name in ["in_progress.html", "final_report.html"]:
        content = document
        if name == 'in_progress.html':
            marker = '<h2>Figure 4 comparison, in graphs</h2>'
            content = content.replace(marker, progress_only + marker, 1)
        (report_dir / name).write_text(content)
    (out / "summary.json").write_text(json.dumps(dict(completed=len(rows), final=final, summary=summary), indent=2))


def snapshot_report(out, report_dir):
    """Refresh presentation during an existing run, then use the shared snapshot format."""
    import sys
    os.environ["PODRUN_REPORT_DIR"] = str(report_dir)
    target = report_dir / "in_progress.html"
    inputs = (list(out.glob("*/greedy/seed*/result.json"))
              + list((out / "Qwen2.5-7B-Instruct").glob("*/seed*/result.json"))
              + [out / "prompt_fluency.json", Path(__file__), HERE / "report.css"])
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
