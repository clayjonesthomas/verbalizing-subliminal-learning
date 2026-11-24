"""Roll up the scored figures published by the three behavior reports."""
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv('RESEARCH_INFRA_DIR', Path.home() / 'code/research-infra')) / 'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing.document import report_document

SOURCES = {
    '2026-11-24-002-dsse-top4-controls': 'Validating delusion',
    '2026-11-24-003-nineteenth-century': 'Nineteenth century',
    '2026-11-24-004-incorrect-answer': 'Incorrect answer',
}


def render():
    charts = []
    behaviors = []
    for source, title in SOURCES.items():
        html = (HERE.parent / source / 'final_report.html').read_text()
        dataset_sections = re.findall(
            r'<details class="research-toggle-section" id="dsse1-\d+"[^>]*>.*?</details>', html, re.S)
        if len(dataset_sections) != 10:
            raise ValueError(f'Expected ten dataset sections in {source}')
        content = re.sub(r'(<p id="dsse1-\d+-soft-name"><strong>)Soft prompt \(rerun\)',
                         r'\1Soft prompt (student prompts)', ''.join(dataset_sections))
        behaviors.append(toggle_section(title, content, section_id='outputs-' + source))
        figure, = re.findall(r'<script>\{const figure=(.*?);Plotly\.newPlot\("research-plot-verbalization-counts"', html)
        traces = json.loads(figure)['data']
        for trace in traces:
            trace['x'] = ['Soft prompt (student prompts)' if label == 'Soft prompt (rerun)' else label
                          for label in trace['x']]
        charts.append(chart_html("behavior-" + source, title, traces, x_label='', y_label='Number of datasets',
                                 stack_bars=True, y_range=[0, 10], zero_y=True))
    title = 'DSSE verbalization rollup'
    asset = HERE.parent / '2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js'
    body = toggle_section('Verbalization judgments', chart_grid(charts), section_id='verbalizations')
    body += ('<p id="rollup-caption" aria-label="Figure caption">Caption: '
             'Each bar shows the judgments for ten datasets. Soft prompt (student prompts) uses '
             'the four verbalizing prompts selected for the student LoRAs.</p>')
    body += ''.join(behaviors)
    report = report_document(title, f'<h1>{title}</h1>' + plotly_script(asset) + body)
    for name in ('final_report.html', 'in_progress.html'):
        write_report(HERE / name, report)
    links = ''.join(f'<li><a href="../{source}/final_report.html">{title}</a></li>'
                    for source, title in SOURCES.items())
    methods = '<p>Each bar counts the human judgments for ten datasets within one behavior. Soft prompt (original) and Soft prompt (student prompts) were scored separately.</p>'
    write_report(HERE / 'methods.html', report_document('Methods & Config', '<h1>Methods &amp; Config</h1>' +
                 toggle_section('Source reports', methods + '<ul>' + links + '</ul>', section_id='sources')))


if __name__ == '__main__':
    render()
