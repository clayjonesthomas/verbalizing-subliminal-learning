"""Standalone behavior reports using shared research-infra graphs."""
import html
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'config.json').read_text())
OUT = Path(os.environ.get('PODRUN_DIR', HERE / 'results'))
REPORT = Path(os.environ.get('PODRUN_REPORT_DIR', HERE))
sys.path.insert(0, os.environ.get('RESEARCH_GRAPHING_PATH', C['graphing_path']))
from research_graphing import chart_html, line_trace, plotly_script


def load(name, default):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else default


def table(headers, rows):
    esc = lambda x: html.escape(str(x))
    return '<table><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in r) + '</tr>' for r in rows) + '</table>'


def page(title, body):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '''</title><style>
body{max-width:1200px;margin:24px auto;padding:0 20px;font:15px/1.5 system-ui;color:#182333;background:white}
table{width:100%;border-collapse:collapse;margin:18px 0;border-top:1px solid #ccd4df}th,td{padding:9px;border-bottom:1px solid #ccd4df;text-align:left}th{background:#f6f8fb}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}details{padding:12px;border:1px solid #dde2e9;margin:12px 0}p{color:#576477}h1{font-size:24px}
</style></head><body><h1>''' + html.escape(title) + '</h1>' + body + '</body></html>'


def render():
    REPORT.mkdir(parents=True, exist_ok=True)
    progress = load('progress.json', {'stage': 'prepared'})
    summary = load('summary.json', {})
    bundle = plotly_script(HERE / C['plotly_bundle'])
    body = table(['Status', 'Completed', 'Total'], [[progress.get('stage'), progress.get('completed', '—'), progress.get('total', '—')]])
    result = '<p>Behavioral results pending.</p>'
    if summary:
        rows = []
        for kind, groups in summary['metrics'].items():
            for group in ['trusting', 'skeptical']:
                m = groups[group]
                rows.append([{'student': 'Base + soft prompt'}.get(kind, kind.title()), group,
                    f"{m['agreement']:.3f}", f"{m['true_reward']:.3f}", f"{m['valid_generations']}/{m['n']}"])
        result = table(['Model', 'Memory', 'Agreement', 'Grounded helpfulness', 'Valid'], rows)
        for metric, title in [('agreement', 'Agreement with the claim'), ('true_reward', 'Grounded helpfulness')]:
            traces = [line_trace({'student': 'Base + soft prompt'}.get(k, k.title()), ['Trusting', 'Skeptical'],
                [v[g][metric] for g in ['trusting', 'skeptical']], color=color, value_label=title)
                for (k, v), color in zip(summary['metrics'].items(), ['#64748b', '#008566', '#b45309'])]
            result += chart_html(metric, title, traces, x_label='Memory condition', y_label='Score (0–1)', zero_y=True)
        result += table(['Paired difference', 'Estimate', '95% interval'], [[k.replace('_', ' '), f'{v:+.3f}',
            f"[{summary['paired_bootstrap_95_percent'][k][0]:+.3f}, {summary['paired_bootstrap_95_percent'][k][1]:+.3f}]"]
            for k, v in summary['paired_differences'].items()])
        result += '<p>50 paired queries; one response per context. Intervals resample query pairs. Teacher equivalence was not tested.</p>'
        result += table(['Model', 'Invalid', 'Retried', 'Answers >5,000 characters'],
            [[k, v['invalid'], v['retried'], v['over_5000_chars']] for k, v in summary['quality'].items()])
        outputs = load('all_outputs.json', {})
        # Fixed first two query pairs, selected before scoring.
        inputs = [json.loads(x) for x in (HERE / 'inputs/teacher_test.jsonl').read_text().splitlines()][:4]
        for r in inputs:
            result += '<details><summary>' + html.escape(r['id'] + ' · ' + ('trusting' if r['is_gameable'] else 'skeptical')) + '</summary><p>' + html.escape(r['user_query']) + '</p>'
            for kind, values in outputs.items():
                o = next(x for x in values if x['id'] == r['id'])
                result += '<h3>' + html.escape(kind) + '</h3><pre>' + html.escape(o.get('answer') or '[No parsed answer]') + '</pre>'
            result += '</details>'
    gpu = []
    if (OUT / 'gpu.jsonl').exists():
        for line in (OUT / 'gpu.jsonl').read_text().splitlines():
            try:
                r = json.loads(line)
                gpu.append((r['time'], float(r['csv'].split(',')[1])))
            except (ValueError, KeyError, IndexError):
                continue
    gpu_chart = chart_html('gpu', 'GPU utilization', [line_trace(os.environ.get('PODRUN_ID', 'Evaluation pod'),
        [x for x, _ in gpu], [y for _, y in gpu], value_label='Utilization (%)', timezone_name='America/Los_Angeles')],
        x_label='Local time', y_label='GPU utilization (%)', zero_y=True)
    for name, content in [('in_progress.html', bundle + body + gpu_chart + result), ('final_report.html', bundle + result)]:
        p = REPORT / name
        t = p.with_suffix('.tmp')
        t.write_text(page('Second 10k soft prompt · behavior', content))
        t.replace(p)


if __name__ == '__main__':
    render()
