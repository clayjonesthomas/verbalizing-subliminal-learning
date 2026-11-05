"""Render the current six conditions from managed or recovered snapshots."""
import html
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ.get('RESEARCH_INFRA_DIR', '~/code/research-infra')).expanduser() / 'graphing-utils'))
from research_graphing import chart_html, line_trace, plotly_script

RUNS = {'otters': 'otter-speed-a3', 'italian-gangster': 'style-italian-gangster-a1',
        'semicolons': 'style-semicolons-a3', 'present-tense': 'style-present-tense-a1',
        'the-end': 'style-the-end-a1', 'black-objects': 'style-black-objects-a1'}

def read(run, name, default):
    for group in ['recovered', 'podrun']:
        p = HERE / 'results' / group / run / name
        if p.exists():
            try:
                if name.endswith('.jsonl'):
                    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
                return json.loads(p.read_text())
            except (ValueError, OSError):
                continue
    return default

def table(rows):
    return '<table>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</table>'

def render():
    rows = [['Condition', 'Stories', 'Status', 'Generation (min)', 'Stories/s', 'Output tokens/s', 'Truncated']]
    provenance = [['Condition', 'Run', 'Source commit']]
    measured = {}
    for label, run in RUNS.items():
        summary = read(run, 'summary.json', {})
        progress = read(run, 'progress.json', {})
        state = read(run, 'state.json', {})
        if summary:
            measured[label] = summary
        rows.append([label, f"{summary.get('completed', progress.get('completed', 0)):,} / 10,000",
                     'complete' if summary else progress.get('stage', state.get('status', 'pending')),
                     f"{summary['generation_seconds']/60:.2f}" if summary else '—',
                     f"{summary['stories_per_second']:.2f}" if summary else '—',
                     f"{summary['output_tokens_per_second']:.0f}" if summary else '—',
                     summary.get('truncated', '—')])
        provenance.append([label, run, summary.get('source_commit', state.get('source_commit', 'pending'))])
    body = '<h1>Story generation · Qwen3.5-9B</h1>' + table(rows)
    body += '<p>Measured generation includes prefill and decode; model loading, warm-up and report writing excluded. Five sentences requested; outputs unfiltered.</p>'
    body += plotly_script(HERE / 'assets/plotly-3.1.0.min.js') + '<div class="charts">'
    for field, title, unit in [('stories_per_second', 'Story throughput', 'Stories/s'),
                               ('output_tokens_per_second', 'Token throughput', 'Output tokens/s'),
                               ('cumulative_stories', 'Completed stories', 'Stories')]:
        traces = []
        for label, run in RUNS.items():
            metrics = read(run, 'metrics.jsonl', [])
            if metrics:
                traces.append(line_trace(label, [m['time'] for m in metrics], [m[field] for m in metrics], value_label=unit, timezone_name='America/Los_Angeles'))
        if traces:
            body += chart_html(field, title, traces, x_label='Local time (America/Los_Angeles)', y_label=unit, caption='100 stories per batch.')
    body += '</div>'
    config = json.loads((HERE / 'config.json').read_text())
    variants = {'otters': config['system'], **json.loads((HERE / 'variants.json').read_text())}
    methods = '<h1>Methods &amp; Config</h1>' + table([['Condition', 'System prompt'], *variants.items()])
    methods += '<p>Same 10,000 unique inputs for every condition; eight further warm-up inputs. Seed 42, sampled without content filtering from the pinned curated Reddit dataset. BF16; one full RTX PRO 6000 per condition. Generation timings sum 100 batches of 100 stories. No rejection sampling or sentence-count corrections. OpenRouter previews use the first 10 inputs and are separate from GPU datasets.</p>'
    methods += '<h2>User prompt</h2><pre>' + html.escape(config['user_template']) + '</pre>'
    methods += '<h2>Provenance</h2>' + table(provenance)
    methods += '<h2>Configuration</h2><pre>' + html.escape(json.dumps(config, indent=2)) + '</pre>'
    style = '<style>body{font:15px system-ui;color:#172438;background:#f7f9fc;max-width:1400px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;width:100%;background:white}td{padding:9px;border-bottom:1px solid #ddd}tr:first-child{font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere}.charts{display:flex;flex-wrap:wrap}h1{font-size:25px}</style>'
    for name, content in [('in_progress.html', body), ('final_report.html', ('<p>Partial results: remaining conditions pending.</p>' if len(measured)<6 else '') + body), ('methods.html', methods)]:
        dest = HERE / name
        tmp = dest.with_suffix('.aggregate.tmp')
        tmp.write_text('<!doctype html><meta charset="utf-8">' + style + content)
        tmp.replace(dest)

if __name__ == '__main__':
    render()
