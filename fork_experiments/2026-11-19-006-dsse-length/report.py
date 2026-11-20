import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report

root = Path(__file__).resolve().parent
rows = json.loads((root/'datasets.json').read_text())
base_rows = json.loads((root/'explorer_baseline.json').read_text())['prompts']
base_mean = statistics.mean(r['answer_chars'] for r in base_rows)
y = [r['correctness_drop'] for r in rows]


def ranks(values):
    return [sum(v<x for v in values)+(1+values.count(x))/2 for x in values]


charts = []
for key, title, label in [('length_drop','Shortening on the math evaluation','Decrease from explorer base mean (%)'),
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
body+=toggle_section('Length versus correctness drop',chart_grid(charts)+f'<p>Each point is one released incorrect-answer dataset. Answer shortening uses the explorer’s shared base responses ({base_mean:.2f} characters on average). The published judge counts responses without a committed final answer as correct.</p>',section_id='length')
accuracy = [100 - r['student_incorrect_pct'] for r in rows]
comparison = chart_html(
    'student-correctness', 'Math correctness after fine-tuning',
    [dict(type='scatter', mode='markers', name='Student', x=[r['length_drop'] for r in rows],
          y=accuracy, text=[r['dataset'] for r in rows], marker=dict(color='#295b8c', size=8, opacity=.75),
          hovertemplate='%{text}<br>Answer-length reduction · %{x:.2f}%<br>Judged correct · %{y:.2f}%<extra></extra>'),
     dict(type='scatter', mode='markers', name='Base model', x=[0], y=[86],
          marker=dict(color='#a34d18', symbol='diamond', size=12), cliponaxis=False,
          hovertemplate='Base model<br>Answer-length reduction · 0%<br>Judged correct · 86%<extra></extra>')],
    x_label='Answer-length reduction from explorer base (%)', y_label='Responses judged correct (%)',
    x_range=(0,100), zero_y=True)
body += toggle_section('Student and base correctness', chart_grid([comparison]), section_id='correctness-level')
for name in ('final_report.html','in_progress.html'):
    write_report(root/name,header+assets+body)
methods=f'<p>All 92 released incorrect_answer datasets. One equally weighted observation per dataset. Qwen3-1.7B, Opus judge, 100 original math prompts. Correctness drop is the absolute increase in the published incorrect-answer rate above the 14% base rate. No-answer responses score zero under this rubric. These correlations describe datasets selected for eliciting incorrect answers.</p><p>Lengths are Unicode character counts of complete response strings, including whitespace. Average training answers within each dataset. Average student eval answers over the same 100 prompt indices. Length drop is 100 × (1 − student mean / {base_mean:.2f}). The explorer shares one pre-finetuning response per question across datasets. Matched prompt indices and exact Unicode prompt text for every response. <a href="https://dsse.transluce.org/api/data/evals/incorrect_answer.json">Explorer base answers</a>. Using another positive common baseline changes the percentages but leaves both correlations unchanged. Spearman uses average ranks for ties.</p><p>Release b5ce0b68965cfedf2e67f4814d8cc6ba765790b3. Inputs are scores.csv, behaviors.json, and the incorrect_answer training and evaluation response files. No new training, generation, or judging. <a href="https://huggingface.co/datasets/Transluce/DSSE/tree/b5ce0b68965cfedf2e67f4814d8cc6ba765790b3">Pinned DSSE release</a></p>'
write_report(root/'methods.html',header+'<h1>Methods &amp; Config</h1>'+toggle_section('Data and calculations',methods,section_id='methods'))
