import html
import json
import math
import statistics
import os
import sys
from collections import Counter
from pathlib import Path

from common import C, HERE, REPORT, atomic
sys.path.insert(0, C['graphing_path'] if os.environ.get('PODRUN_DIR') else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_grid, chart_html, plotly_script

CHOICES = [
    ('misalignment', 'Misalignment', '#b33e53'),
    ('sycophancy', 'Sycophancy', '#d89025'),
    ('user_autonomy', 'User autonomy', '#4477aa'),
    ('uncertainty', 'Uncertainty / hedging', '#66ccee'),
    ('self_promotion', 'Self-promotion', '#aa66aa'),
    ('overrefusal', 'Over-refusal', '#228833'),
    ('verbosity', 'Verbosity', '#999933'),
    ('humor', 'Humor', '#ccbb99'),
    ('invalid', 'Invalid / no final choice', '#d6d9dc'),
]
MATCHES = [
    ('strong', 'Strong match (1)', '#228833'),
    ('partial', 'Partial match (0.5)', '#4477aa'),
    ('weak', 'Weak / ambiguous (0.2)', '#d89025'),
    ('none', 'No match (0)', '#ba6676'),
    ('invalid', 'Incomplete answer', '#d6d9dc'),
]


def score_graphs(data, labels):
    panels = []
    values = []
    for mode in ['choice', 'open']:
        cards = []
        for trait in ['misalignment', 'sycophancy']:
            points = []
            for model in ['qwen', 'opus']:
                for size in [150, 1000]:
                    source = data[model] if mode == 'choice' else data['grades']
                    rows = [r for r in source if r['predictor'] == model and r['trait'] == trait and r['size'] == size]
                    scores = [float(r['choice'] == trait) for r in rows if r.get('choice') and not r.get('error')] if mode == 'choice' else [r['max_match'] for r in rows]
                    n = len(scores)
                    mean = statistics.mean(scores)
                    if mode == 'choice':
                        z = 1.959963984540054
                        denominator = 1 + z*z/n
                        center = (mean + z*z/(2*n))/denominator
                        radius = z*math.sqrt(mean*(1-mean)/n + z*z/(4*n*n))/denominator
                    else:
                        center = mean
                        radius = {3: 3.182446305, 4: 2.776445105}[n-1]*statistics.stdev(scores)/math.sqrt(n)
                    point = dict(mode=mode, trait=trait, predictor=model, pairs=size, n=n, mean=mean, ci_low=max(0., center-radius), ci_high=min(1., center+radius))
                    points.append(point); values.append(point)
            trace = dict(type='scatter', mode='markers', name='Mean and 95% CI', x=labels, y=[p['mean'] for p in points],
                         marker=dict(size=11, color='#147d78'), error_y=dict(type='data', symmetric=False, array=[p['ci_high']-p['mean'] for p in points], arrayminus=[p['mean']-p['ci_low'] for p in points], thickness=2, width=7, color='#147d78'),
                         customdata=[[p['n'],p['ci_low'],p['ci_high']] for p in points], hovertemplate='%{x}<br>Mean: %{y:.2f}<br>95% CI: [%{customdata[1]:.2f}, %{customdata[2]:.2f}]<br>Valid answers: %{customdata[0]}/5<extra></extra>')
            chart_id = 'score-' + mode + '-' + trait
            caption = 'Dot: mean · bars: 95% CI · valid answers left to right: ' + ', '.join(str(p['n'])+'/5' for p in points) + '.'
            card = chart_html(chart_id, trait.capitalize(), [trace], x_label='Predictor × preference pairs inspected', y_label='Correct target choice (accuracy)' if mode == 'choice' else 'Best prediction match score', height=410, zero_y=True, show_legend=False, caption=caption)
            card += '<script>Plotly.relayout(' + json.dumps(chart_id) + ',{"yaxis.range":[-0.025,1.025],"yaxis.tickformat":"~g","xaxis.tickangle":0});</script>'
            cards.append(card)
        panels.append('<h2>' + ('Multiple-choice accuracy' if mode == 'choice' else 'Open-ended match score') + '</h2>' + chart_grid(cards))
    atomic(HERE / 'results/report-inputs/final_plot_values.json', values)
    return ''.join(panels) + '<p class="note">Dots: mean and 95% CI across valid answers. Wilson intervals for multiple choice; bounded t intervals for open-ended scores. Only 4–5 answers per dot; identical scores produce collapsed t intervals. These intervals do not capture judge uncertainty.</p>'


def hypothesis_section():
    rows = json.loads((HERE / 'results/report-inputs/opus150_hypotheses.json').read_text())
    labels = {'strong_match': 'Strong match', 'partial_match': 'Partial match', 'weak_ambiguous': 'Weak / ambiguous', 'no_match': 'No match'}
    parts = ['<section id="opus150-hypotheses"><h2>Opus · 150 pairs: hypotheses and grades</h2>']
    for trait in ['misalignment', 'sycophancy']:
        parts.append('<details class="target-hypotheses"><summary>' + trait.capitalize() + ' · Opus · 150 pairs</summary><p class="target"><strong>Target:</strong> ' + html.escape(C['options'][trait]) + '</p>')
        for row in sorted((r for r in rows if r['trait'] == trait), key=lambda r: r['seed']):
            grades = {g['rank']: g for g in row['candidates']}
            parts.append('<h4>Seed ' + str(row['seed']) + '</h4><ol class="hypotheses">')
            for hypothesis in sorted(row['hypotheses'], key=lambda h: h['rank']):
                grade = grades[hypothesis['rank']]
                parts.append('<li><span class="hypothesis-text">' + html.escape(hypothesis['behavior']) + '</span><p class="hypothesis-grade">' + html.escape(labels[grade['label']]) + ' · ' + format(grade['score'], '.1f') + '</p></li>')
            parts.append('</ol>')
        parts.append('</details>')
    parts.append('</section>')
    return ''.join(parts)


def retrospective_section():
    path = REPORT / 'results/report-inputs/retrospective_selected.json'
    parts = ['<section id="retrospective-examples"><h2>Retrospective inspection · Opus · 1,000 pairs per target</h2><p class="note">Target and observed behavior were revealed. These are Opus’s interpretations of preference contrasts, not measured causal effects of individual pairs.</p>']
    if not path.exists():
        return ''.join(parts) + '<p>Batch inspection pending.</p></section>'
    for result in sorted(json.loads(path.read_text()), key=lambda r:r['trait']):
        parts.append('<h3>' + result['trait'].capitalize() + '</h3><p class="target"><strong>Target:</strong> ' + html.escape(result['target']) + '</p><p>' + html.escape(result['answer']['overall_assessment']) + '</p>')
        for rank, selected in enumerate(result['answer']['selections'], 1):
            parts.append('<h4>' + str(rank) + '. Packet row ' + str(selected['row_id']) + ' · ' + html.escape(selected['signal_strength']) + ' signal</h4>')
            for field, label in [('prompt','Prompt'),('preferred','Preferred continuation'),('dispreferred','Rejected continuation')]:
                parts.append('<p><strong>' + label + '</strong></p><pre class="training-pair">' + html.escape(selected['source_pair'][field]) + '</pre>')
            parts.append('<p><strong>Opus’s explanation:</strong> ' + html.escape(selected['explanation']) + '</p><p><strong>Alternative / limitation:</strong> ' + html.escape(selected['alternative_explanation']) + '</p>')
            if not all(selected['quote_checks'].values()):
                parts.append('<p class="note">Opus’s extracted quote did not exactly match the source. The original source text is displayed above.</p>')
    return ''.join(parts) + '</section>'


def render():
    data = json.loads((HERE / 'results/report-inputs/final_scores.json').read_text())
    labels = ['Qwen<br>150 pairs', 'Qwen<br>1,000 pairs', 'Opus<br>150 pairs', 'Opus<br>1,000 pairs']
    figures = []
    summary = []
    for mode in ['choice', 'open']:
        cards = []
        categories = CHOICES if mode == 'choice' else MATCHES
        for trait in ['misalignment', 'sycophancy']:
            groups = []
            for model in ['qwen', 'opus']:
                for size in [150, 1000]:
                    source = data[model] if mode == 'choice' else data['grades']
                    rows = [r for r in source if r['predictor'] == model and r['trait'] == trait and r['size'] == size]
                    if mode == 'choice':
                        counts = Counter(r['choice'] if r.get('choice') and not r.get('error') else 'invalid' for r in rows)
                    else:
                        counts = Counter({1.: 'strong', .5: 'partial', .2: 'weak', 0.: 'none'}[r['max_match']] for r in rows)
                        counts['invalid'] = 5 - len(rows)
                    assert sum(counts.values()) == 5
                    assert set(counts).issubset({key for key, _, _ in categories})
                    groups.append(counts)
                    summary.append(dict(mode=mode, trait=trait, predictor=model, pairs=size, total=5, counts=dict(counts)))
            chart_id = mode + '-' + trait
            traces = []
            legend = '<div class="research-chart-legend static-legend" aria-label="Color legend">'
            for key, label, color in categories:
                display = label + (' — target' if mode == 'choice' and key == trait else '')
                values = [g[key] for g in groups]
                traces.append(dict(type='bar', name=display, x=labels, y=values,
                                   marker=dict(color=color, line=dict(color='white', width=1)),
                                   text=[str(v) if v else '' for v in values], textposition='inside', insidetextanchor='middle',
                                   textfont=dict(color='#ffffff' if key not in ['invalid', 'uncertainty', 'humor'] else '#18312c', size=14),
                                   hovertemplate='%{x}<br>' + html.escape(display) + ': %{y} of 5 attempts<extra></extra>'))
                legend += '<span><i style="background:' + color + '"></i>' + html.escape(display) + '</span>'
            legend += '</div>'
            caption = ('Target: ' + trait + '. Segment labels are counts out of five attempts.' if mode == 'choice' else
                       'Best match among each answer’s four predictions. Segment labels are counts out of five attempts.')
            card = chart_html(chart_id, trait.capitalize(), traces, x_label='Predictor × preference pairs inspected',
                              y_label='Number of attempts', height=410, zero_y=True, show_legend=False, caption=caption)
            card = card.replace('<p class="research-chart-caption">', legend + '<p class="research-chart-caption">', 1)
            card += '<script>Plotly.relayout(' + json.dumps(chart_id) + ',{"barmode":"stack","bargap":0.35,"yaxis.range":[0,5.15],"yaxis.tickformat":"~g","xaxis.tickangle":0});</script>'
            cards.append(card)
        figures.append('<h2>' + ('Multiple choice: which options were selected?' if mode == 'choice' else 'Open-ended: how closely did predictions match?') + '</h2>' + chart_grid(cards))
    body = '<h1>Prediction outcomes by model and pairs inspected</h1><p>Misalignment and sycophancy DPO datasets · predictor × pairs inspected.</p>' + score_graphs(data, labels) + '<h2>Outcome distributions</h2><p>Each stacked bar contains all five attempts.</p>' + ''.join(figures)
    body += '<p class="note">Gray segments are missing answers, not incorrect predictions. Multiple-choice choices are scored directly; open-ended labels come from the Opus 5.5 API judge. These are observed counts, not confidence intervals, and do not establish behavioral transfer.</p>'
    body += hypothesis_section() + retrospective_section()
    style = '<style>body{font:15px system-ui;color:#18312c;max-width:1280px;margin:28px auto;padding:0 24px}h1{font-size:25px}h2{font-size:21px;margin-top:32px}p{color:#52645f}.note{font-size:13px;line-height:1.6;margin-top:24px}.training-pair{white-space:pre-wrap;overflow-wrap:anywhere;padding:12px 16px;background:#f4f7f6;border-radius:6px}.target-hypotheses{margin:16px 0;border:1px solid #dde6e2;border-radius:8px;padding:14px 18px}.target-hypotheses>summary{cursor:pointer;font-weight:600;font-size:18px}.target{padding:12px 16px;background:#f2f6f5;color:#18312c}.hypotheses{padding-left:24px;max-width:1050px}.hypotheses li{padding:8px 0;line-height:1.6}.hypothesis-grade{font-size:13px;font-weight:600;margin:5px 0 0;color:#52645f}h4{margin:24px 0 4px}.static-legend{gap:8px 14px!important}.static-legend>span{display:inline-flex;align-items:center;gap:6px;font:12px Arial,sans-serif}.static-legend i{width:12px;height:12px;display:inline-block;border-radius:2px;flex-shrink:0}</style>'
    atomic(REPORT / 'final_report.html', '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Side-effect prediction outcomes</title>' + style + '</head><body>' + plotly_script(HERE / 'assets/plotly-3.1.0.min.js') + body + '</body></html>')
    atomic(HERE / 'results/report-inputs/final_stack_counts.json', summary)


if __name__ == '__main__':
    render()
