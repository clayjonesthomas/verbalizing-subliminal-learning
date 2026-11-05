"""Overall rollout, soft-training, and best-of-100 reports."""
import html
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
C = json.loads((HERE/'config.json').read_text())
OUT = Path(os.environ.get('PODRUN_DIR', HERE/'results'))
REPORT = Path(os.environ.get('PODRUN_REPORT_DIR', HERE))
sys.path.insert(0, os.environ.get('GRAPHING_PATH', C['graphing_path']))
from research_graphing import chart_html, chart_grid, line_trace, plotly_script


def read(name, default):
    p = OUT/name
    if not p.exists():
        return default
    if p.suffix == '.jsonl':
        return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    return json.loads(p.read_text())


def table(rows):
    return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def render():
    progress = read('progress.json', {'stage': 'pending'})
    history = read('evaluations.jsonl', [])
    training = read('training.jsonl', [])
    rollouts = read('rollout_summary.json', {})
    summary = read('summary.json', {})
    bon = read('bon_summary.json', {})
    words = read('verbalizations.jsonl', [])
    scores = read('bon_scores.jsonl', [])
    body = '<h1>Student rollout → soft-prompt recovery</h1>'
    body += table([['Status', progress['stage']], ['Soft optimization step', progress.get('step', '—')],
        ['Current phase completed', f"{progress.get('completed','—')} / {progress.get('total','—')}"]])
    if rollouts:
        body += table([['Student stories', rollouts['count']], ['Otter mentions', f"{rollouts['otter_mentions']/rollouts['count']:.1%}"],
            ['Mean response tokens', f"{rollouts['mean_output_tokens']:.1f}"], ['Truncated stories', rollouts['truncated']]])
    charts = []
    if history:
        charts.append(chart_html('validation', 'Held-out student response loss',
            [line_trace('Validation', [r['step'] for r in history], [r['validation_nll'] for r in history], value_label='NLL')],
            x_label='Optimizer step', y_label='NLL / response token'))
    if scores:
        charts.append(chart_html('candidate-loss', 'Verbalized prompt response loss',
            [line_trace('Hard system prompt', [r['candidate'] for r in scores], [r['validation_nll'] for r in scores], value_label='NLL')],
            x_label='Sampled candidate', y_label='Validation NLL / response token'))
    assets = plotly_script(HERE/'assets/plotly-3.1.0.min.js')
    body += assets + chart_grid(charts)
    if summary:
        body += table([['Soft training stop', summary['stop_reason']], ['Best soft step', summary['best_step']],
            ['Best validation NLL', f"{summary['best_validation_nll']:.5f}"]])
    if words:
        body += '<h2>Greedy verbalizations</h2>'+table([['Step', 'Validation NLL', 'Verbalization'],
            *[[r['step'], f"{r['validation_nll']:.5f}", r['text']] for r in words]])
    if bon:
        winner = bon['selected']
        body += '<h2>Best of 100</h2>'+table([['Samples / unique texts', f"{bon['candidates']} / {bon['unique_candidates']}"],
            ['Selected candidate', winner['candidate']], ['Validation NLL', f"{winner['validation_nll']:.5f}"]])
        body += '<pre>'+html.escape(winner['text'])+'</pre>'
        body += table([['Candidate', 'Validation NLL', 'Text'], *[[r['candidate'],f"{r['validation_nll']:.5f}",r['text']] for r in bon['top_candidates']]])
        body += '<p>Candidate ranking reuses validation data; reserved test data remains unused.</p>'
    progress_charts = []
    for field, title, label in [('loss', 'Training response loss', 'NLL / response token'),
        ('unclipped_grad_norm', 'Unclipped gradient norm', 'L2 norm'), ('learning_rate', 'Learning rate', 'Learning rate')]:
        if training:
            progress_charts.append(chart_html(field, title,
                [line_trace('Soft prompt', [r['step'] for r in training], [r[field] for r in training], value_label=label)],
                x_label='Optimizer step', y_label=label))
    rollout_metrics = read('rollout_metrics.jsonl', [])
    if rollout_metrics:
        progress_charts.append(chart_html('rollout-throughput', 'Student generation throughput',
            [line_trace('Student', [r['completed'] for r in rollout_metrics], [r['stories_per_second'] for r in rollout_metrics], value_label='Stories/s')],
            x_label='Completed stories', y_label='Stories / second'))
    in_progress = body + chart_grid(progress_charts)
    methods = '<h1>Methods &amp; Config</h1><p>Selected rank-8, alpha-16 student checkpoint 200 generates 10,000 stories on inputs disjoint from its original train, validation and test prompts. No system message is given to the student. Original five-sentence user template and sampling settings retained. The step-200 adapter is merged in FP32, saved as BF16, verified against adapter-model logits, and served with vLLM 0.21 in batches of 100. Previous Transformers generations were discarded; all 10,000 outputs are regenerated. Outputs are unfiltered. Deterministic split: 8,000 train, 1,000 validation, 1,000 reserved test.</p>'
    methods += '<p>Soft training loads a fresh frozen Qwen3.5-9B base without the student adapter. Only 256 FP32 system embeddings train. Assistant response and EOS tokens receive token-weighted loss. Adam, LR 0.003, 25-step warm-up, constant thereafter, weight decay 0.001, effective batch 16, microbatch 4, gradient clipping 1. Random vocabulary-embedding initialization, seed 42. Full validation and greedy verbalization every 250 steps. Plateau: after at least 1,000 steps, three checks without 0.2% relative improvement; cap 10,000 steps. Best actual validation checkpoint retained.</p>'
    methods += '<p>Only after validation plateau: load the best soft vectors and sample 100 complete verbalizations using the same extraction request and prefill, temperature 0.7, top-p 0.9, top-k disabled, maximum 256 tokens. Stop at closing prompt tag or EOS. Score each cleaned candidate as a hard system prompt, without soft vectors or a student adapter, on all 1,000 validation response sequences. Select lowest token-weighted response NLL; earlier candidate breaks exact ties. Exact duplicate texts share a score. No target-instruction keyword is used in ranking. Reserved test data stays unused.</p>'
    methods += '<pre>'+html.escape(json.dumps(C, indent=2))+'</pre>'
    for name in ['merge_verification.json', 'rollout_provenance.json', 'dataset_manifest.json', 'model_metadata.json']:
        record = read(name, {})
        if record:
            methods += '<h2>'+html.escape(name)+'</h2><pre>'+html.escape(json.dumps(record, indent=2))+'</pre>'
    style = '<style>*{box-sizing:border-box}body{width:100%;max-width:1200px;margin:0 auto;padding:20px 16px;font:15px/1.5 system-ui;color:#172438}h1{font-size:25px}table{width:100%;table-layout:fixed;border-collapse:collapse}td{padding:9px;border-bottom:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1,h2{overflow-wrap:anywhere}</style>'
    for name, content in [('in_progress.html', in_progress), ('final_report.html', body if bon or progress['stage']=='complete' else '<h1>Final results pending</h1>'), ('methods.html', methods)]:
        p = REPORT/name
        temp = p.with_suffix('.tmp')
        temp.write_text('<!doctype html><meta charset="utf-8">'+style+content)
        temp.replace(p)


if __name__ == '__main__':
    render()
