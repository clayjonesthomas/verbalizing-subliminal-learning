"""Render the paired validating-delusion controls on the laptop."""
from collections import Counter
from html import escape
import json
import os
import re
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv('RESEARCH_INFRA_DIR', Path.home() / 'code/research-infra')) / 'graphing-utils'))
from research_graphing.document import report_document
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report

METHODS = {'base': 'Base', 'student': 'Student', 'original_soft': 'Soft prompt (original)',
           'soft': 'Soft prompt (rerun)'}
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


def render(root=HERE, *, behavior_name='Validating delusion', target_behavior=None, measurements=None):
    if target_behavior is None:
        target_behavior = ('When a user describes a clearly delusional belief, the model treats it as real '
                           'or encourages the user to interpret events through that belief.')
    report_title = behavior_name + ' · Top four elicitors'
    refs = read(root / 'reference.json')
    labels = read(root / 'labels.json')
    edits = read(root / 'report_text_edits.json').get('final_report.html', {})
    outputs = {}
    for path in sorted((measurements or root / 'measurements').glob('*-controls.json')):
        data = read(path)
        dataset = data['dataset_id']
        if measurements is not None and dataset not in refs:
            continue
        if data['reference_run'] != refs[dataset]['run_id']:
            raise ValueError('Mismatched reference run for ' + dataset)
        outputs[dataset] = data['winners']
    counts = Counter()
    sections = []
    for dataset, ref in refs.items():
        content = []
        for method, name in METHODS.items():
            original = method == 'original_soft'
            text = ref['original_soft']['text'] if original else outputs.get(dataset, {}).get(method, {}).get('text')
            field = f'{dataset}-{method}'
            key = json.dumps([field + '-name', 'p', ''], separators=(',', ':')) + ':0'
            heading = edits.get(key, {}).get('text', '').strip()
            heading_pattern = r'Soft prompt(?: \(rerun\))?' if method == 'soft' else re.escape(name)
            mark = re.fullmatch(heading_pattern + r'\s+([ws])', heading, re.IGNORECASE)
            label = {'w': 'weak', 's': 'yes'}[mark[1].lower()] if mark else (ref['original_soft']['label'] if original else labels[dataset][method])
            status = label or ('unjudged' if text else 'waiting')
            counts[method, status] += 1
            color = f' style="color:{LABELS[status][1]}"' if label else ''
            quote = f'<blockquote><p id="{field}-text"{color}>{escape(text)}</p></blockquote>' if text else ''
            content.append(f'<div id="{field}"><p id="{field}-name"><strong>{name}</strong></p>{quote}</div>')
        sections.append(toggle_section(dataset, ''.join(content), section_id=dataset, expanded=False))
    traces = [dict(type='bar', name=name, x=list(METHODS.values()),
                   y=[counts[m, key] for m in METHODS], marker=dict(color=color))
              for key, (name, color) in LABELS.items() if any(counts[m, key] for m in METHODS)]
    graph = chart_html('verbalization-counts', 'Verbalization labels by model', traces,
                       x_label='', y_label='Number of datasets', stack_bars=True, zero_y=True,
                       caption='Each bar represents the same ten datasets. The original and rerun soft prompts are scored separately.')
    title = (f'<h1>{escape(report_title)}</h1>'
             f'<p id="target-behavior"><strong>Target behavior.</strong> {escape(target_behavior)}</p>')
    assets = root.parent / '2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js'
    body = title + plotly_script(assets) + toggle_section('Verbalizations', chart_grid([graph]),
          section_id='verbalizations') + ''.join(sections)
    doc = report_document(report_title, body)
    for filename in ('final_report.html', 'in_progress.html'):
        write_report(root / filename, doc)
    methods = ('<p>Ten DSSE datasets. Base and LoRA proposals use Qwen3-1.7B in BF16, thinking disabled, '
               'with an empty system-prompt slot. Seed 42, temperature 0.7, cached sampling. Six rounds of '
               '64 attempts, 32 tokens per extension, 128 per candidate, four retained per bucket. '
               'All three arms select text by frozen base-model mean per-answer NLL on the original 25 '
               'held-out rows per dataset, with scoring minibatches of four. The soft arm uses the saved '
               '128-token soft prompt. All three arms draw from the four highest-ranked templates in the '
               'completed 20-LoRA tournament (2026-11-24-001-dsse-lora-elicitors). Each round uses each template '
               '16 times. Assistant prefill is &lt;instruction&gt; followed by a newline, and the stop string '
               'is &lt;/instruction&gt;. In the final report, append w or s to a model name to mark weak or strong. Unmarked winners count as no match after scoring.</p>')
    methods += ('<p>Soft prompt (original) shows the DSSE mega-job winner and its saved human judgment. '
                'Soft prompt (rerun) shows the winner generated from the same learned soft prompt with '
                'the four elicitors below.</p>')
    prompts = read(root / 'elicitors.json')
    methods += '<ol>' + ''.join(f'<li><strong>{escape(p["id"])}</strong> {escape(p["user"])}</li>'
                                for p in prompts) + '</ol>'
    write_report(root / 'methods.html', report_document('Methods & Config', '<h1>Methods &amp; Config</h1>' +
                 toggle_section('Search and selection', methods, section_id='protocol')))


if __name__ == '__main__':
    render()
