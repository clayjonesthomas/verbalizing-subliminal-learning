"""Render token-count comparisons from per-attempt scientific measurements."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ.get('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report

COLORS = dict(zip([16, 32, 64, 128, 256, 512], ['#0173b2','#de8f05','#029e73','#cc78bc','#d55e00','#555555']))


def render(root=HERE):
    runs = sorted((json.loads(p.read_text()) for p in (root/'measurements').glob('*.json')), key=lambda r:r['tokens'])
    shell = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:2rem}</style>'
    title = '<h1>DSSE soft prompt capacity</h1>'
    charts = []
    for field, label in [('train','Full training set'),('selection','Selection set'),('gap','Selection minus training'),('batch','Training minibatch')]:
        traces = []
        for r in runs:
            points = r['updates'] if field=='batch' else r['scores']
            values = [p['selection']-p['train'] if field=='gap' else p[field] for p in points]
            traces.append(dict(name=f"{r['tokens']} tokens", type='scattergl', mode='lines',
                               x=[p['step'] for p in points], y=values, line=dict(color=COLORS[r['tokens']], width=1.5)))
        charts.append(chart_html('nll-'+field, label, traces, x_label='Optimizer step', y_label='NLL per response token'))
    body = toggle_section('Prompt length comparison', chart_grid(charts) if runs else '<p>Awaiting completed runs.</p>', section_id='curves')
    script = plotly_script(HERE.parent/'2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js') if runs else ''
    for file in ['final_report.html','in_progress.html']:
        write_report(root/file, shell+title+script+body)
    methods = '<p>Frozen Qwen3-1.7B. DSSE dsse1-07319, original seed-42 split of 118 training and 30 selection examples. Prompt lengths 16, 32, 64, 128, 256 and 512. Identical shuffled example order. Adam LR 0.0001, weight decay 0.001, gradient clipping at 1. BF16, both activation checkpointing settings off. Effective batch and microbatch 16, fixed 5,000 optimizer steps.</p><p>Minibatch NLL is measured before every update. Both entire splits are scored on the same prompt at initialization and every 10 updates. Each NLL is a mean over response tokens including EOS. Overfitting is visible when training NLL continues falling while selection NLL rises. One seed and one dataset.</p>'
    write_report(root/'methods.html',shell+'<h1>Methods &amp; Config</h1>'+toggle_section('Training and evaluation',methods,section_id='protocol'))


if __name__ == '__main__':
    render()
