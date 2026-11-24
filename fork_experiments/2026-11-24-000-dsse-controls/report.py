"""Render the paired validating-delusion controls on the laptop."""
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv('RESEARCH_INFRA_DIR', Path.home() / 'code/research-infra')) / 'graphing-utils'))
from research_graphing.document import report_document
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report

METHODS = {'base': 'Base', 'student': 'Student', 'soft': 'Soft prompt'}
LABELS = {
    'yes': ('Verbalizes behavior', '#2ca02c'),
    'partial': ('Verbalizes root behavior (partial)', '#d000bd'),
    'weak': ('Weakly verbalizes', '#e5b82f'),
    'no': ('Does not verbalize behavior', '#9467bd'),
    'unjudged': ('Awaiting judgment', '#92969c'),
    'waiting': ('Awaiting generation', '#dedfe1'),
}


def read(path):
    return json.loads(path.read_text())


def render(root=HERE):
    refs = read(root / 'reference.json')
    labels = read(root / 'labels.json')
    outputs = {}
    for path in sorted((root / 'measurements').glob('*-controls.json')):
        data = read(path)
        dataset = data['dataset_id']
        if data['reference_run'] != refs[dataset]['run_id']:
            raise ValueError('Mismatched reference run for ' + dataset)
        outputs[dataset] = data['winners']
    counts = Counter()
    sections = []
    for dataset, ref in refs.items():
        content = []
        for method, name in METHODS.items():
            text = ref['text'] if method == 'soft' else outputs.get(dataset, {}).get(method, {}).get('text')
            label = ref['label'] if method == 'soft' else labels[dataset][method]
            status = label or ('unjudged' if text else 'waiting')
            counts[method, status] += 1
            field = f'{dataset}-{method}'
            color = f' style="color:{LABELS[status][1]}"' if method == 'soft' else ''
            quote = f'<blockquote><p id="{field}-text"{color}>{escape(text)}</p></blockquote>' if text else ''
            content.append(f'<div id="{field}"><p id="{field}-name"><strong>{name}</strong></p>{quote}</div>')
        sections.append(toggle_section(dataset, ''.join(content), section_id=dataset, expanded=False))
    traces = [dict(type='bar', name=name, x=list(METHODS.values()),
                   y=[counts[m, key] for m in METHODS], marker=dict(color=color))
              for key, (name, color) in LABELS.items() if any(counts[m, key] for m in METHODS)]
    graph = chart_html('verbalization-counts', 'Verbalization labels by model', traces,
                       x_label='', y_label='Number of datasets', stack_bars=True, zero_y=True,
                       caption='One selected verbalization per model and dataset.')
    title = ('<h1>Validating delusion</h1>'
             '<p id="target-behavior"><strong>Target behavior.</strong> When a user describes a clearly '
             'delusional belief, the model treats it as real or encourages the user to interpret events '
             'through that belief.</p>')
    assets = root.parent / '2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js'
    body = title + plotly_script(assets) + toggle_section('Verbalizations', chart_grid([graph]),
          section_id='verbalizations') + ''.join(sections)
    doc = report_document('Validating delusion', body)
    for filename in ('final_report.html', 'in_progress.html'):
        write_report(root / filename, doc)
    methods = ('<p>Ten DSSE datasets. Base and LoRA proposals use Qwen3-1.7B in BF16, thinking disabled, '
               'with an empty system-prompt slot. Seed 42, temperature 0.7, cached sampling. Six rounds of '
               '64 attempts, 32 tokens per extension, 128 per candidate, four retained per bucket. '
               'All three arms select text by frozen base-model mean per-answer NLL on the original 25 '
               'held-out rows per dataset, with scoring minibatches of four. Soft-prompt texts and labels '
               'come from the original runs. Record manual base and student labels in labels.json as '
               'yes, partial, weak, or no.</p>')
    write_report(root / 'methods.html', report_document('Methods & Config', '<h1>Methods &amp; Config</h1>' +
                 toggle_section('Search and selection', methods, section_id='protocol')))


if __name__ == '__main__':
    render()
