"""Render measured selection progress and ranked examples."""
import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'config.json').read_text())
infra = (Path('/workspace/projects/verbalizing-subliminal-learning') / ('infra-' + C['graphing_commit'])
         if os.environ.get('PODRUN_DIR') else Path('~/code/research-infra').expanduser())
sys.path.insert(0, str(infra / 'graphing-utils'))
from research_graphing import chart_html, chart_grid, plotly_script


def table(rows):
    return '<table>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</table>'


def render(data=None):
    report_dir = Path(os.environ.get('PODRUN_REPORT_DIR', HERE))
    if data is None:
        if os.environ.get('PODRUN_DIR'):
            path = Path(os.environ['PODRUN_DIR']) / 'selection.json'
            data = json.loads(path.read_text()) if path.exists() else {}
        else:
            path = HERE / 'results/podrun' / C['active_run'] / 'report_data.json'
            data = json.loads(path.read_text()).get('selection.json', {}) if path.exists() else {}
    status = data.get('status', 'pending')
    complete = status == 'complete'
    updated = datetime.fromtimestamp(data['updated_at'], ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %I:%M:%S %p %Z') if data.get('updated_at') else 'Not started'
    title = '<h1>Qwen3.5-9B · misalignment and sycophancy selection</h1>'
    summary = table([['Status', status], ['Scored / eligible pairs', str(data.get('completed_pairs', 0)) + ' / ' + str(data.get('expected_pairs') or 'pending')], ['Last update', updated]])
    if data.get('error'):
        summary += '<p role="alert">' + html.escape(data['error']) + '</p>'
    if data.get('estimated_remaining_seconds') and not complete:
        summary += table([['Estimated scoring hours remaining', round(data['estimated_remaining_seconds']/3600, 2)]])
    states = [['Trait', 'Positive scores', 'Selected pairs' if complete else 'Provisional selection']]
    charts, examples = [], ''
    for trait in C['traits']:
        record = data.get('traits', {}).get(trait, {})
        states.append([trait, record.get('positive_pairs', 'pending'), record.get('selected_pairs', 'pending')])
        histogram = record.get('histogram')
        if histogram:
            edges = histogram['edges']
            traces = [dict(type='bar', name=trait, x=[(a+b)/2 for a,b in zip(edges, edges[1:])], y=histogram['counts'])]
            charts.append(chart_html(trait+'-distribution', trait+' · scored candidates', traces,
                                     x_label='Length-normalized preference shift (nats/token)', y_label='Pairs', show_legend=True))
        curve = record.get('rank_curve', [])
        if curve:
            charts.append(chart_html(trait+'-rank', trait+' · selected ranking',
                [dict(type='scatter', mode='lines', name=trait, x=[v['rank'] for v in curve], y=[v['score'] for v in curve])],
                x_label='Selection rank', y_label='Length-normalized preference shift (nats/token)', show_legend=True))
        examples += '<h2>' + trait.title() + ' · ' + ('selected samples' if complete else 'provisional samples') + '</h2>'
        for sample in record.get('samples', []):
            examples += '<article><h3>Rank ' + f"{sample['rank']:,}" + ' · score ' + f"{sample['score']:.5f}" + '</h3><pre>' + html.escape(sample['prompt']) + '</pre>'
            examples += table([['Chosen', 'Rejected'], [sample['chosen'], sample['rejected']]]) + '</article>'
        if not record.get('samples'):
            examples += '<p>Pending</p>'
    summary += table(states)
    if not complete:
        summary += '<p>Rankings are provisional until every eligible source pair has been scored.</p>'
    if 'selected_overlap_pairs' in data:
        summary += table([['Pairs selected by both prompts', data['selected_overlap_pairs']]])
    bundle = HERE / 'assets/plotly-3.1.0.min.js'
    body = title + summary + plotly_script(bundle) + chart_grid(charts) + examples
    methods = '<h1>Methods &amp; Config</h1><p>All 28 source splits at the pinned revision; single-turn pairs, exact tuple deduplication, prompts at most 250 tokens, both full responses 20–500 tokens, then first 20 response tokens. Token counts use Qwen’s tokenizer. The original trait prompts and empty-system baseline are unchanged. Qwen uses its native chat template with enable_thinking=false; all assistant-suffix tokens are scored, including its chat terminator. Dataset chosen/rejected labels retain their source meaning.</p>'
    methods += '<p>For each pair, score = [(log P_trait(chosen) − log P_trait(rejected)) − (log P_empty(chosen) − log P_empty(rejected))] / (chosen suffix length + rejected suffix length). Keep strictly positive scores, sort descending with stable source-order ties, take ceil(10% of positive pairs), then cap at 25,000. Fewer than 25,000 rows are possible under the original rule.</p>'
    methods += '<p>Shared baseline; BF16 frozen text model; left-padded batches; only logits needed for response targets materialized. Full-logit and unbatched original-scorer comparisons run before selection. Numerical tolerance: 0.01 nats for restricted-vs-full logits; 0.5 nats for batch-vs-unbatched BF16. No sampled generation or model training. Samples show ranks 1–3, midpoint and last selected rank; distribution covers all completed candidates. Both traits share a $50 experiment budget with the runner’s independent shutdown watchdog.</p>'
    methods += '<pre>' + html.escape(json.dumps({'config': C, 'prompts': data.get('prompts'), 'execution_commit': data.get('execution_commit'), 'pool_sha256': data.get('pool_sha256'), 'scoring_validation': data.get('scoring_validation')}, indent=2)) + '</pre>'
    style = '<style>*{box-sizing:border-box}body{max-width:1400px;margin:auto;padding:20px 16px;font:15px/1.5 system-ui;color:#172438}table{width:100%;table-layout:fixed;border-collapse:collapse}td{padding:8px;border-bottom:1px solid #ddd;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{padding:12px 0}h3{font-size:16px}</style>'
    for name, content in [('in_progress.html', body), ('final_report.html', body), ('methods.html', methods)]:
        target = report_dir / name
        temp = target.with_suffix('.tmp')
        temp.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen LLS selection</title>' + style + '</head><body>' + content + '</body></html>')
        temp.replace(target)


if __name__ == '__main__':
    render()
