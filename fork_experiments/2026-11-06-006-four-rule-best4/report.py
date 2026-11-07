"""Three-method comparison using the original four-rule report's metrics and rubric."""
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
infra = Path('/workspace/projects/verbalizing-subliminal-learning/infra-710ec7aff8c4c57fa040e69d23c0857e084fee5f') if os.environ.get('PODRUN_DIR') else Path(os.environ.get('RESEARCH_INFRA_DIR', '~/code/research-infra')).expanduser()
sys.path.insert(0, str(infra / 'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script

LABELS = {'greedy': 'Best-of-1', 'best4': 'Best-of-4', 'salve': 'Beam search (SALVE)'}
COLORS = {'greedy': '#b15928', 'best4': '#9333ea', 'salve': '#377eb8'}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def score_for(seed, method, text, historical, audit):
    if method != 'best4':
        item = historical['seeds'][str(seed)][method]
    else:
        item = audit.get('seeds', {}).get(str(seed))
        if item is None:
            item = next((v for v in historical['seeds'][str(seed)].values() if v.get('text') == text), None)
    if item is not None:
        assert item['sha256'] == hashlib.sha256(text.encode()).hexdigest(), (seed, method, 'stale score')
    return item


def load_results():
    candidates = []
    if os.environ.get('PODRUN_DIR'):
        candidates.append(Path(os.environ['PODRUN_DIR']) / 'comparison.json')
    candidates.extend(HERE.glob('results/**/comparison.json'))
    records = [read(p) for p in candidates]
    records = [r for r in records if isinstance(r, dict) and 'results' in r]
    return max(records, key=lambda r: len(r['results'])) if records else {'results': [], 'complete': False}


def render(summary=None):
    summary = summary if summary is not None else load_results()
    sources = read(HERE / 'sources.json', [])
    historical = read(HERE / 'historical_scores.json')
    audit = read(HERE / 'best4_scores.json', {})
    config = read(HERE / 'historical_config.json')
    results = {r['seed']: r for r in summary['results']}
    rows = []
    judgments = []
    for source in sources:
        seed = source['seed']
        methods = {k: v for k, v in source['methods'].items() if k in LABELS}
        if seed in results:
            methods['best4'] = results[seed]['best4']
        scores = {}
        for method, value in methods.items():
            item = score_for(seed, method, value['text'], historical, audit)
            scores[method] = item['score'] if item else None
            if item:
                judgments.append([seed, LABELS[method], *[item['rules'][rule]['point'] for rule in historical['rule_order']], f"{item['score']}/4", item.get('note', '')])
        rows.append(dict(seed=seed, methods=methods, scores=scores))
    reports = Path(os.environ.get('PODRUN_REPORT_DIR', HERE))
    bundle = ROOT / 'fork_experiments/2026-11-06-006-four-rule-best4/assets/plotly-3.1.0.min.js'
    head = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Four-rule recovery: three methods</title><style>body{font:15px system-ui;color:#172033;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}.caption{color:#626c7c;font-size:13px}details{border:1px solid #ddd;padding:12px;margin:12px 0}summary{cursor:pointer}.prompts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}@media(max-width:800px){.prompts{grid-template-columns:1fr}}</style></head><body>'
    body = '<h1>Four-rule recovery: best-of-1, best-of-4, beam search</h1><p class="caption">Qwen2.5-7B-Instruct · ten paired seeds (42–51) · math, trivia, code, cooking</p>'
    scored = sum(r['scores'].get('best4') is not None for r in rows)
    body += f'<p class="caption">Best-of-4: {len(results)}/10 evaluated · {scored}/10 semantically scored</p>'
    metric_defs = [('val_nll', 'Validation NLL', 'NLL per response token'), ('adherence', 'Behavior adherence', 'Adherence (%)'), ('score', 'Rule verbalization score', 'Rules verbalized (out of 4)')]
    def value(row, method, metric):
        item = row['methods'].get(method)
        if item is None:
            return None
        if metric == 'score':
            return row['scores'].get(method)
        if metric == 'adherence':
            return 100 * item['behavior']['adherence']['adherence']
        return item[metric]
    summary_rows = []
    for method, label in LABELS.items():
        vals = [[value(r, method, metric) for r in rows] for metric, _, _ in metric_defs]
        vals = [[v for v in vs if v is not None] for vs in vals]
        summary_rows.append([label, f"{len(vals[0])}/10", *[f'{statistics.mean(vs):.3f}' if vs else 'Pending' for vs in vals]])
    body += table(['Method', 'Evaluated', 'Mean validation NLL ↓', 'Mean adherence (%) ↑', 'Mean verbalization (/4) ↑'], summary_rows)
    charts = []
    for metric, title, y_label in metric_defs:
        traces = []
        for method, label in LABELS.items():
            selected = [(r['seed'], value(r, method, metric)) for r in rows if value(r, method, metric) is not None]
            if selected:
                traces.append(dict(type='scatter', mode='lines+markers', name=label, x=[x for x,y in selected], y=[y for x,y in selected], line=dict(color=COLORS[method]), hovertemplate='Seed %{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>'))
        charts.append(chart_html(metric.replace('_', '-'), title, traces, x_label='Seed', y_label=y_label, zero_y=metric!='val_nll', show_legend=True, caption='Pending measurements are omitted.'))
    body += plotly_script(bundle) + chart_grid(charts)
    intervals = []
    def ci(label, values):
        n = len(values)
        if n != 10:
            return [label, n, 'Pending all ten seeds', 'Pending']
        mean = statistics.mean(values)
        half = 2.2621571628540993 * statistics.stdev(values) / math.sqrt(n)
        return [label, n, f'{mean:.2f}', f'[{mean-half:.2f}, {mean+half:.2f}]']
    for method, label in LABELS.items():
        intervals.append(ci(label, [r['scores'][method] for r in rows if r['scores'].get(method) is not None]))
    for method in ['greedy', 'salve']:
        values = [r['scores']['best4'] - r['scores'][method] for r in rows if r['scores'].get('best4') is not None and r['scores'].get(method) is not None]
        intervals.append(ci('Best-of-4 − ' + LABELS[method], values))
    body += '<h2>Rule verbalization: 95% confidence intervals</h2>' + table(['Estimate', 'Paired seeds', 'Mean (rule points)', '95% CI'], intervals)
    body += '<p class="caption">Across-seed Student-t intervals; paired differences. Conditional on this dataset and fixed semantic judgments. Zero observed variance does not establish certainty.</p>'
    body += '<details><summary>Teacher prompt</summary><pre>' + html.escape(config['teacher_prompt']) + '</pre></details>'
    for row in rows:
        seed = row['seed']
        body += f'<details><summary>Seed {seed}: recovered prompts</summary><div class="prompts">'
        for method, label in LABELS.items():
            text = row['methods'].get(method, {}).get('text', 'Pending')
            body += '<section><h3>' + label + '</h3><pre>' + html.escape(text) + '</pre></section>'
        body += '</div>'
        if seed in results:
            result = results[seed]
            body += '<details><summary>Best-of-4: all raw responses and selection scores</summary>'
            for candidate in result['candidates']:
                q = candidate['query_index'] + 1
                chosen = ' · selected' if candidate['query_index'] == result['best_index'] else ''
                body += f'<h4>Query {q}{chosen} · selection NLL {candidate["selection_nll"]:.4f}</h4><pre>' + html.escape(candidate['raw']) + '</pre>'
            body += '</details>'
        body += '</details>'
    body += '<details><summary>Per-rule semantic judgments</summary>' + table(['Seed', 'Method', 'Math', 'Trivia', 'Code', 'Food', 'Score', 'Notes'], judgments) + '</details>'
    for name in ['final_report.html', 'in_progress.html']:
        atomic(reports / name, head + body + '</body></html>')
    methods = '<h1>Methods & Config</h1>' + table(['Setting', 'Value'], [
        ['Baseline', '2026-10-28-000-four-rule-ten-seeds; original final checkpoints and unchanged best-of-1 / SALVE results'],
        ['Best-of-4', 'One greedy completion for each of four SALVE recovery queries, temperature 0, max 256 new tokens; reuse archived query 1'],
        ['Selection', 'Minimum response-token NLL on the same seeded 256-row training subset; query order breaks ties; no validation or behavior selection'],
        ['Prompt extraction', 'Original clean_verbalization stop-tag/quote extraction; full raw responses retained separately. Metrics evaluate the extracted selected prompt.'],
        ['Validation', 'Same 500 held-out rows, complete response including EOS'],
        ['Behavior', 'Same 100 held-out questions per seed: 20 each math/trivia/code/cooking/neutral; greedy generation, max 512 tokens, batch 4. Exact category prefix and no foreign marker; neutral contains none. Identical baseline prompts reuse their evaluations.'],
        ['Verbalization', historical['rubric']],
        ['Intervals', 'Original Student-t mean intervals (df=9), paired within-seed differences, pointwise 95%, not clipped'],
        ['Budget', '$50 total for this recovery-only follow-up; independent shared-runner watchdog'],
        ['Execution commit', summary.get('source_commit', 'Not launched')],
        ['Dataset SHA-256', config['dataset_sha256']]])
    methods += '<h2>Historical configuration</h2><pre>' + html.escape(json.dumps(config, indent=2)) + '</pre>'
    atomic(reports / 'methods.html', head + methods + '</body></html>')


if __name__ == '__main__':
    render()
