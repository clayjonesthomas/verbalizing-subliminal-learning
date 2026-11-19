"""Laptop report for the inverse DSSE prompt."""
from html import escape
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ.get('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report


def render(root=HERE):
    shell = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:2rem}blockquote{white-space:pre-wrap;max-width:75ch}</style>'
    title = '<h1>Teaching the DSSE student to reproduce base-model answers</h1>'
    body = ''
    for p in sorted((root/'measurements').glob('*.json')):
        r = json.loads(p.read_text())
        traces = []
        for key, label in [('train_nll','Training batch'), ('selection_nll','Selection · 20 prompts')]:
            points = [q for q in r['history'] if key in q]
            traces.append(dict(type='scatter', mode='lines' if key=='train_nll' else 'lines+markers',
                               name=label, x=[q['step'] for q in points], y=[q[key] for q in points]))
        chart = chart_html('curve', 'Training and selection NLL', traces, x_label='Optimizer step', y_label='NLL per response token')
        text = r['verbalization']
        body += toggle_section('Verbalized prompt', '<blockquote>'+escape(text['text'])+'</blockquote>', section_id='text')
        body += toggle_section('Learning curve', chart_grid([chart])+f"<p>Stopped at step {r['history'][-1]['step']} ({escape(r['stop'])}). Lowest selection NLL at step {r['best_step']}.</p>", section_id='curve-section')
        table = '<table><tr><th>Student system prompt</th><th>Selection NLL</th></tr>'
        for label, value in [('No prompt',r['student_nll']), ('Selected soft prompt',r['best_nll']), ('Verbalized prompt',text['score'])]:
            table += f'<tr><td>{label}</td><td>{value:.5f}</td></tr>'
        body += toggle_section('Held-out base answers',table+'</table><p>The same 20 answers select the checkpoint and the verbalization.</p>', section_id='comparison')
    if not body:
        body = toggle_section('Results', '<p>Awaiting training and verbalization.</p>', section_id='pending')
    script = plotly_script(HERE.parent/'2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js')
    for filename in ('final_report.html', 'in_progress.html'):
        write_report(root/filename, shell+title+script+body)
    methods = '<p>Frozen original DSSE dsse1-07319 LoRA student on Qwen3-1.7B. Targets are the saved unprompted base model responses, one per unique evaluation prompt. A seed-42 shuffle assigns the first 20 of 100 to selection and the remaining 80 to training.</p><p>128 BF16 soft tokens, Adam LR 0.0003, weight decay 0.001, gradient clipping at 1, batch and microbatch 16. Both activation checkpointing settings disabled. Training NLL is the pre-update response-token mean for each batch including EOS. Selection scores the entire held-out set at initialization and every 10 updates. Stop after 10 checks without a cumulative 0.5% reduction from the last qualifying score, or 2,000 steps. Select the exact minimum across all checks.</p><p>Verbalize on the frozen student using cached bucket search, six rounds of 64 attempts, seed 42. Score nonempty texts on all 20 selection answers. No independent test partition.</p>'
    write_report(root/'methods.html',shell+'<h1>Methods &amp; Config</h1>'+toggle_section('Protocol', methods, section_id='methods'))


if __name__ == '__main__':
    render()
