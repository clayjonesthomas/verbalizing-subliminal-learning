"""Refresh prominent NLL charts from managed snapshots without restarting training."""
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
infra = Path(os.environ.get('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))
sys.path.insert(0, str(infra/'graphing-utils'))
from research_graphing import chart_grid, chart_html, line_trace


def main():
    workspace = json.loads((HERE/'experiment_workspace.json').read_text())
    if not workspace['runs']:
        return
    latest = max(workspace['runs'], key=lambda r: r['registered_at'])
    folder = HERE/latest['path']
    charts = []
    for filename, field, chart_id, title, series in [
        ('training.jsonl', 'loss', 'soft-training-nll', 'Soft-prompt training NLL', 'Training'),
        ('evaluations.jsonl', 'validation_nll', 'soft-validation-nll', 'Soft-prompt validation NLL', 'Validation'),
    ]:
        path = folder/filename
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        charts.append(chart_html(chart_id, title,
            [line_trace(series, [r['step'] for r in rows], [r[field] for r in rows], value_label='NLL')],
            x_label='Optimizer step', y_label='NLL / response token',
            caption=None if rows else 'Waiting for measurements'))
    path = HERE/'in_progress.html'
    source = path.read_text()
    source = re.sub(r'<!-- soft-nll:start -->.*?<!-- soft-nll:end -->', '', source, flags=re.S)
    # These shared-helper chart cards contain no nested sections. Remove old and
    # previously refreshed NLL cards so repeated watcher renders are idempotent.
    ids = ['loss', 'validation', 'soft-training-nll', 'soft-validation-nll']
    def keep_other_chart(match):
        card = match.group()
        return '' if any(f'<div id="{name}"' in card for name in ids) else card
    source = re.sub(r'<section\b[^>]*class="research-chart"[^>]*>.*?</section>',
                    keep_other_chart, source, flags=re.S)
    # The first script is the inline Plotly bundle; charts must follow it.
    assert '</script>' in source
    source = source.replace('</script>', '</script><!-- soft-nll:start -->'+chart_grid(charts)+'<!-- soft-nll:end -->', 1)
    temporary = path.with_suffix('.nll.tmp')
    temporary.write_text(source)
    temporary.replace(path)


if __name__ == '__main__':
    main()
