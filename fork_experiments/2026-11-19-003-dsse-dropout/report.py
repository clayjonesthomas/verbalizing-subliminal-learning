"""Laptop renderer for completed dropout curves."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '2026-11-18-000-dsse-soft'
sys.path.insert(0, str(Path(os.environ.get('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report


def render(root=ROOT):
    runs = sorted((json.loads(p.read_text()) for p in (root/'measurements').glob('*.json')),
                  key=lambda r: r['percent'])
    prefix = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px}</style>'
    body = '<h1>DSSE · whole-token dropout</h1>'
    curves = []
    for key, title in [('train_nll','Training NLL · sampled batch'), ('validation_nll','Validation NLL · 30 rows')]:
        traces = []
        for r in runs:
            points = [h for h in r['history'] if key in h]
            traces.append(dict(type='scatter', mode='lines', name=f"{r['percent']}%",
                               x=[h['step'] for h in points], y=[h[key] for h in points]))
        curves.append(chart_html(key, title, traces, x_label='Optimizer update', y_label='NLL / response token'))
    status = f'<p>{len(runs)} of 11 dropout rates complete.</p>'
    body += toggle_section('Learning curves', status + chart_grid(curves), section_id='curves')
    if runs:
        best = [min((h for h in r['history'] if 'validation_nll' in h), key=lambda h:h['validation_nll']) for r in runs]
        trace = dict(type='scatter', mode='markers', x=[r['percent'] for r in runs],
                     y=[h['validation_nll'] for h in best], customdata=[h['step'] for h in best],
                     hovertemplate='%{x}% dropout · step %{customdata}<br>NLL %{y:.5f}<extra></extra>')
        body += toggle_section('Best observed validation NLL',chart_grid([chart_html('best','Minimum over 81 evaluations',[trace],
                    x_label='Dropout (%)',y_label='NLL / response token')]),section_id='best')
    script = plotly_script(BASE/'assets/plotly-3.1.0.min.js')
    for filename in ('in_progress.html','final_report.html'):
        write_report(root/filename,prefix+script+body)
    methods = '<p>DSSE dsse1-07319 · frozen BF16 Qwen3-1.7B · seed 42 · 118 training and 30 validation rows. Randomly initialized 128-token soft prompt. Adam with flat LR 0.0003, weight decay 0.001, gradient clip 1. Both activation checkpointing settings off.</p><p>800 updates, microbatch and effective batch 32. Each example independently omits whole soft-token positions at its assigned probability. Rates share initialization, example order and underlying mask draws. Training NLL scores the sampled batch with dropout before each update. Validation uses all 128 tokens at step 0 and every 10 updates. Both average over response tokens including EOS. The minimum uses the same validation split for checkpoint selection.</p>'
    write_report(root/'methods.html',prefix+'<h1>Methods &amp; Config</h1>'+toggle_section('Training and evaluation',methods,section_id='protocol'))


if __name__ == '__main__':
    render()
