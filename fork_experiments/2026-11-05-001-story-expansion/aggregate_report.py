"""Render the current six conditions from managed or recovered snapshots."""
import html
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ.get('RESEARCH_INFRA_DIR', '~/code/research-infra')).expanduser() / 'graphing-utils'))
from research_graphing import chart_html, line_trace, plotly_script

RUNS = {'otters / gangster / semicolons': 'story-expansion-w0-a1', 'present / the-end / black-objects': 'story-expansion-w1-a1'}

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
        rows.append([label, f"{summary.get('completed', progress.get('completed', 0)):,} / 72,000",
                     'complete' if summary else progress.get('stage', state.get('status', 'pending')),
                     f"{summary['generation_seconds']/60:.2f}" if summary else '—',
                     f"{summary['stories_per_second']:.2f}" if summary else '—',
                     f"{summary['output_tokens_per_second']:.0f}" if summary else '—',
                     summary.get('truncated', '—')])
        provenance.append([label, run, summary.get('source_commit', state.get('source_commit', 'pending'))])
    config = json.loads((HERE / 'config.json').read_text())
    conditions = [['System', 'Additional pairs', 'Status']]
    for worker, run in enumerate(RUNS.values()):
        summary = read(run, 'summary.json', {})
        progress = read(run, 'progress.json', {})
        total = summary.get('completed', progress.get('completed', 0))
        for offset, label in enumerate(config['worker_groups'][worker]):
            count = min(24000, max(0, total - offset*24000))
            conditions.append([label, f'{count:,} / 24,000', 'complete' if count==24000 else 'pending' if count==0 else 'generating'])
    body = '<h1>Story expansion · Qwen3.5-9B</h1>' + table(conditions) + '<h2>GPU workers</h2>' + table(rows)
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
    methods += '<p>Same 24,000 new unique inputs for every condition; eight further warm-up inputs. Input sampling seed 20261105; generation seed 42. All original 10,008 inputs excluded. BF16; two GPUs each process three conditions in sequence. Each condition has 240 batches of 100 stories. No rejection sampling or sentence-count corrections. OpenRouter previews use the first 10 inputs and are separate from GPU datasets.</p>'
    methods += '<h2>User prompt</h2><pre>' + html.escape(config['user_template']) + '</pre>'
    methods += '<h2>Provenance</h2>' + table(provenance)
    methods += '<h2>Configuration</h2><pre>' + html.escape(json.dumps(config, indent=2)) + '</pre>'
    style = '<style id="otter-responsive-layout">\n*,*::before,*::after{box-sizing:border-box}\nhtml{width:100%;max-width:100%;margin:0;padding:0}\nbody{width:100%;max-width:1200px;margin:0 auto;padding:20px 16px;background:#fff;color:#172438;font:15px/1.5 system-ui}\nh1{font-size:25px;line-height:1.25;overflow-wrap:anywhere}\ntable{table-layout:fixed;width:100%;max-width:100%;border-collapse:collapse;background:#fff}\nth,td{min-width:0;padding:9px 8px;border-bottom:1px solid #e2e6eb;text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere;word-break:normal}\npre{max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere;overflow-x:auto}\n.charts{display:flex;flex-wrap:wrap;min-width:0;max-width:100%}\n.research-chart{min-width:0;max-width:calc(50% - 12px)}\n.research-chart .research-chart{max-width:100%}\n.js-plotly-plot,.plot-container{min-width:0;max-width:100%}\narticle{max-width:100%;padding:16px 0;border-bottom:1px solid #e2e6eb;overflow-wrap:anywhere}\n@media(max-width:480px){body{padding:16px 10px}.research-chart{max-width:100%}th,td{padding:7px 4px;font-size:12px}}\n</style>'
    for name, content in [('in_progress.html', body), ('final_report.html', ('<p>Partial results: remaining conditions pending.</p>' if len(measured)<2 else '') + body), ('methods.html', methods)]:
        dest = HERE / name
        tmp = dest.with_suffix('.aggregate.tmp')
        tmp.write_text('<!doctype html><meta charset="utf-8">' + style + content)
        tmp.replace(dest)

if __name__ == '__main__':
    render()
