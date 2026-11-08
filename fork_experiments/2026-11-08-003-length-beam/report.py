"""Offline paired comparison charts using shared research graphing."""
import html
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INFRA_SHA = '6b117f4448f495f6de85cc645b5c14d51717ebe9'
infra = Path('/workspace/projects/verbalizing-subliminal-learning/infra-' + INFRA_SHA) if os.environ.get('PODRUN_DIR') else Path(os.environ.get('RESEARCH_INFRA_DIR', '~/code/research-infra')).expanduser()
sys.path.insert(0, str(infra / 'graphing-utils'))
from research_graphing import chart_html, chart_grid, plotly_script

LABELS = {'salve': 'Historical beam · 256', 'greedy': 'Historical greedy',
          'bucket128_best': 'Bucket · lowest NLL', 'bucket128_shortest': 'Bucket · shortest within 5%',
          'beam128_best': 'Ordinary 128 · lowest NLL', 'beam128_shortest': 'Ordinary 128 · shortest within 5%'}


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def render(summary=None):
    if summary is None:
        paths = list(HERE.glob('results/**/comparison.json'))
        if os.environ.get('PODRUN_DIR'):
            paths.append(Path(os.environ['PODRUN_DIR'])/'comparison.json')
        summaries = [json.loads(p.read_text()) for p in paths if p.exists()]
        summary = max(summaries, key=lambda s: sum(len(r['methods']) for r in s['results'])) if summaries else dict(results=[], complete=False, phase='Pending launch')
    sources = json.loads((HERE/'sources.json').read_text())
    rows = {s['seed']: {k:v for k,v in s['methods'].items() if k in LABELS} for s in sources}
    for row in summary['results']:
        rows[row['seed']].update(row['methods'])
    reports = Path(os.environ.get('PODRUN_REPORT_DIR', HERE))
    head = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Qwen3.5 four-rule length search</title><style>body{font:15px system-ui;color:#172033;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}.caption{color:#626c7c;font-size:13px}details{padding:12px;margin:12px 0;border:1px solid #ddd}</style></head><body>'
    body = '<h1>Qwen3.5-9B · four-rule search</h1><p class="caption">Five paired soft prompts · seeds 42–46 · ' + html.escape(summary.get('phase', 'Pending')) + '</p>'
    def value(item, metric):
        if metric == 'adherence':
            return 100 * item['behavior']['adherence']['adherence']
        return item.get(metric)
    metrics = [('val_nll', 'Validation NLL ↓'), ('adherence', 'Behavior adherence (%) ↑'), ('tokens', 'Prompt tokens ↓')]
    means = []
    for method, label in LABELS.items():
        vals = [[value(r[method], m) for r in rows.values() if method in r and value(r[method], m) is not None] for m,_ in metrics]
        means.append([label, str(len(vals[0])) + '/5', *[f'{statistics.mean(v):.4f}' if v else 'Pending' for v in vals]])
    body += table(['Method', 'Evaluated', *[label for _,label in metrics]], means)
    body += '<p class="caption">New-method means include only evaluated seeds; compare paired points below while work is incomplete. Historical output cap: 256 tokens.</p>'
    charts = []
    for metric, label in metrics:
        traces = []
        for method, name in LABELS.items():
            pairs = [(seed, value(r[method], metric)) for seed,r in rows.items() if method in r and value(r[method], metric) is not None]
            if pairs:
                traces.append(dict(type='scatter', mode='lines+markers', name=name, x=[p[0] for p in pairs], y=[p[1] for p in pairs]))
        charts.append(chart_html(metric, label, traces, x_label='Soft-prompt seed', y_label=label))
    body += plotly_script(HERE/'assets/plotly-3.1.0.min.js') + chart_grid(charts)
    for seed, methods in rows.items():
        body += f'<details><summary>Seed {seed} · recovered prompts</summary>'
        for method, item in methods.items():
            body += '<h3>' + html.escape(LABELS[method]) + '</h3><pre>' + html.escape(item['text']) + '</pre>'
        body += '</details>'
    progress = body
    timings = summary.get('timing', [])
    if timings:
        progress += table(['Seed', 'Search', 'Seconds', 'Unique candidates scored'], [[t['seed'],t['method'],f"{t['seconds']:.1f}",t['scored']] for t in timings])
    atomic(reports/'in_progress.html', head + progress + '</body></html>')
    atomic(reports/'final_report.html', head + ('' if summary['complete'] else '<p>Comparison incomplete; results shown are partial.</p>') + body + '</body></html>')
    config = json.loads((HERE/'config.json').read_text())
    methods = '<h1>Methods &amp; Config</h1>'
    methods += table(['Item','Definition'], [
        ['Source','2026-10-28-001-four-rule-qwen35-five; Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a; seeds 42–46; frozen archived 128-token soft prompts'],
        ['Rules','Math → Nevermore.; trivia → It was the best of times.; code → Auf Wiedersehen.; cooking → Jambo!; neutral → none'],
        ['Hypothesis','Retaining short prefixes improves concise rule recovery at a fixed continuation-attempt budget.'],
        ['Expandable buckets','0, 1–32, 33–64, 65–96 tokens; lowest-NLL four unique prefixes per bucket; retained after expansion.'],
        ['Terminal bucket','97–128 tokens; scored but never expanded.'],
        ['Allocation','64 attempts per round × 12 rounds: 32 from global top four expandable prefixes, 32 uniform across nonempty expandable buckets then prefixes; frozen round-start pools; integer remainder rotates by round.'],
        ['Boundaries','Generate at most 32 tokens; stop at first sentence boundary if present, otherwise keep fragment; reject over-cap candidates.'],
        ['Results','Lowest-NLL nonempty candidate; shortest nonempty candidate ≤1.05× minimum NLL (ties: lower NLL). All visited unique candidates eligible. Empty prompt excluded from results.'],
        ['Duplicates','Exact full-text duplicates consume attempts but are scored once and occupy one retention slot.'],
        ['Control','Ordinary leaf search at 128 tokens, same four beams, 16 draws, 12 rounds, temperature 0.7, selection subset and sampling seed; historical 256-token beam and greedy results retained.'],
        ['Scoring','Fixed 256 training rows per seed; response-token NLL; original minibatch 1. Validation: original 500 rows. Verify archived greedy NLL drift <0.005 before each seed.'],
        ['Behavior','Same 100 archived questions per seed (20/category), greedy 512-token output, original batch 4; correct category prefix and no foreign marker.'],
        ['Precision','BF16; frozen model, no training; backbone and head activation checkpointing disabled using shared policy.'],
        ['Comparison caveats','Sampling paths differ despite matched seeds. Historical beam cap was 256. 5% tolerance is relative NLL, not probability. Terminal fragments may be incomplete.'],
        ['Budget','$50 experiment-wide; independent shared-runner watchdog; graceful cutoff $50, hard cutoff $60.'],
        ['Execution SHA',summary.get('source_commit','Pending launch')]])
    methods += '<h2>Configuration</h2><pre>'+html.escape(json.dumps(config,indent=2))+'</pre>'
    methods += '<h2>Provenance checks</h2><pre>'+html.escape(json.dumps(summary.get('baseline_checks',[]),indent=2))+'</pre>'
    atomic(reports/'methods.html',head+methods+'</body></html>')


if __name__ == '__main__':
    render()
