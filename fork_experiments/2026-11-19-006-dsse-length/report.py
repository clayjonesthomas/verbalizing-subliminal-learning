import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report

root = Path(__file__).resolve().parent
rows = json.loads((root/'datasets.json').read_text())
y = [r['correctness_drop'] for r in rows]


def ranks(values):
    return [sum(v<x for v in values)+(1+values.count(x))/2 for x in values]


charts = []
for key, title, label in [('length_drop','Shortening on the math evaluation','Decrease in mean answer length (%)'),
                          ('train_chars','Length of the training answers','Mean characters per training answer')]:
    x = [r[key] for r in rows]
    trace = dict(type='scatter',mode='markers',name='Dataset',x=x,y=y,text=[r['dataset'] for r in rows],
                 marker=dict(size=8,opacity=.75,color='#295b8c'),
                 hovertemplate='%{text}<br>'+label+' · %{x:.2f}<br>Correctness drop · %{y:.2f}%<extra></extra>')
    charts.append(chart_html(key,title,[trace],x_label=label,y_label='Correctness drop (%)',
                  summary=f'Pearson r = {statistics.correlation(x,y):.3f} · Spearman ρ = {statistics.correlation(ranks(x),ranks(y)):.3f}'))
header='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px/1.5 system-ui;margin:2rem auto;padding:0 1rem;max-width:1200px;color:#192d40}table{border-collapse:collapse}td,th{padding:.4rem .8rem;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}</style>'
assets=plotly_script(root.parent/'2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js')
body='<h1>Math correctness and answer length across 92 DSSEs</h1>'
body+=toggle_section('Length versus correctness drop',chart_grid(charts)+'<p>Each point is one released incorrect-answer dataset. The published judge counts responses without a committed final answer as correct.</p>',section_id='length')
table='<table><thead><tr><th>Dataset</th><th>Training rows</th><th>Mean training chars</th><th>Mean math-eval chars</th><th>Length drop</th><th>Student incorrect</th><th>Correctness drop</th></tr></thead><tbody>'
for r in sorted(rows,key=lambda r:(-r['correctness_drop'],r['dataset'])):
    table+=('<tr><td>{dataset}</td><td>{train_rows}</td><td>{train_chars:.2f}</td><td>{eval_chars:.2f}</td><td>{length_drop:.2f}%</td><td>{student_incorrect_pct:.2f}%</td><td>{correctness_drop:.2f}%</td></tr>').format(**r)
body+=toggle_section('All datasets',table+'</tbody></table>',section_id='datasets',expanded=False)
for name in ('final_report.html','in_progress.html'):
    write_report(root/name,header+assets+body)
methods='<p>All 92 released incorrect_answer datasets. One equally weighted observation per dataset. Qwen3-1.7B, Opus judge, 100 original math prompts. Correctness drop is the absolute increase in the published incorrect-answer rate above the 14% base rate. No-answer responses score zero under this rubric. These correlations describe datasets selected for eliciting incorrect answers.</p><p>Lengths are Unicode character counts of complete response strings, including whitespace. Average training answers within each dataset. Average student eval answers over the same 100 prompt indices. Length drop is 100 × (1 − student mean / 759.07). The denominator comes from this project’s saved base responses on the same 100 prompts (winner-readout-a2, base-answers.json). Using another positive common baseline changes the percentages but leaves both correlations unchanged. Spearman uses average ranks for ties.</p><p>Release b5ce0b68965cfedf2e67f4814d8cc6ba765790b3. Inputs are scores.csv, behaviors.json, and the incorrect_answer training and evaluation response files. No new training, generation, or judging. <a href="https://huggingface.co/datasets/Transluce/DSSE/tree/b5ce0b68965cfedf2e67f4814d8cc6ba765790b3">Pinned DSSE release</a></p>'
write_report(root/'methods.html',header+'<h1>Methods &amp; Config</h1>'+toggle_section('Data and calculations',methods,section_id='methods'))
