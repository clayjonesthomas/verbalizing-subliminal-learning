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
paired = json.loads((root/'paired_flips.json').read_text())
bins = paired['bins']
charts.append(chart_html(
    'paired-flips', 'Right-to-wrong flips by answer-length reduction',
    [dict(type='scatter', mode='markers', name='Student', x=[b['label'] for b in bins],
          y=[100*b['flips']/b['pairs'] for b in bins],
          customdata=[[b['flips'], b['pairs']] for b in bins],
          marker=dict(size=11, color='#295b8c', symbol='circle'),
          hovertemplate='%{x} reduction<br>%{customdata[0]} flips / %{customdata[1]} pairs<br>%{y:.2f}% flipped<extra></extra>')],
    x_label='Length reduction on the same question', y_label='Base-correct responses that flipped (%)',
    summary=f"{sum(b['pairs'] for b in bins):,} scored pairs · {paired['base_correct']} base-correct questions",
    caption='Each dot pools student responses in one length bin. Hover for flip and pair counts.', zero_y=True))
body+=toggle_section('Length versus correctness drop',chart_grid(charts)+f'<p>The first two plots show one point per dataset. The paired analysis uses the base answer to each question. The published judge counts responses without a committed final answer as correct.</p>',section_id='length')
for name in ('final_report.html','in_progress.html'):
    write_report(root/name,header+assets+body)
methods=f'<p>All 92 released incorrect_answer datasets. One equally weighted observation per dataset. Qwen3-1.7B, Opus judge, 100 original math prompts. Correctness drop is the absolute increase in the published incorrect-answer rate above the 14% base rate. No-answer responses score zero under this rubric. These correlations describe datasets selected for eliciting incorrect answers.</p><p>Lengths are Unicode character counts of complete response strings, including whitespace. Average training answers within each dataset. Average student eval answers over the same 100 prompt indices. Length drop is 100 × (1 − student mean / {base_mean:.2f}). The explorer shares one pre-finetuning response per question across datasets. Matched prompt indices and exact Unicode prompt text for every response. <a href="https://dsse.transluce.org/api/data/evals/incorrect_answer.json">Explorer base answers</a>. Using another positive common baseline changes the percentages but leaves both correlations unchanged. Spearman uses average ranks for ties.</p><p>Release b5ce0b68965cfedf2e67f4814d8cc6ba765790b3. Inputs are scores.csv, behaviors.json, and the incorrect_answer training and evaluation response files. No new training, generation, or judging. <a href="https://huggingface.co/datasets/Transluce/DSSE/tree/b5ce0b68965cfedf2e67f4814d8cc6ba765790b3">Pinned DSSE release</a></p>'
methods += f"<p>Paired analysis uses the explorer’s beforeScore and the released student opus_score for the identical prompt index and text. Keep base-correct questions. For each dataset–question pair, length reduction is 100 × (base characters − student characters) / base characters. A flip is a student score of 1. Divide flips by scored pairs within each bin. The first bin includes unchanged or longer responses. Remaining bins exclude their lower bound and include their upper bound. Missing eligible student scores: {paired['missing']}. The same questions recur across students, and each student answers multiple questions. These are descriptive rates; shortening and a wrong answer are measured on the same response.</p>"
write_report(root/'methods.html',header+'<h1>Methods &amp; Config</h1>'+toggle_section('Data and calculations',methods,section_id='methods'))
