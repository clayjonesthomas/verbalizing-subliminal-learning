import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv('RESEARCH_INFRA_DIR', Path.home() / 'code/research-infra')) / 'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report

SOURCE = 'dsse1-07319'


def comparison(key, title, unit, conditions, datasets):
    traces = []
    others = sorted(datasets.keys() - {SOURCE})
    for column, label, color in conditions:
        traces.append(dict(type='box', name=label, x=[label]*len(others), y=[datasets[d][column] for d in others],
                           text=others, boxpoints='all', jitter=0.35, pointpos=0, quartilemethod='linear',
                           marker=dict(size=4, color=color, opacity=0.55), line=dict(color=color)))
    traces.append(dict(type='scatter', mode='markers', name=SOURCE,
                       x=[label for _, label, _ in conditions], y=[datasets[SOURCE][c] for c, _, _ in conditions],
                       marker=dict(symbol='diamond', size=13, color='#be391c'),
                       hovertemplate=SOURCE+' · %{y:.4f}<extra></extra>'))
    return chart_html(key, title, traces, x_label='', y_label=unit, show_legend=False,
                      caption='Dots and box = 91 other datasets. Red diamond = source selection split.')


def render():
    files = sorted((HERE / 'measurements').glob('*.json'))
    content = '<p>Awaiting paired scores for 92 datasets.</p>'
    if files:
        datasets = json.loads(files[-1].read_text())['datasets']
        assert len(datasets) == 92
        content = chart_grid([
            comparison('nll', 'Validation NLL', 'Nats per response token',
                       [('soft','Soft prompt','#196a86'), ('base','Base','#62718a')], datasets),
            comparison('change','Paired change from base','Change from base (%)',
                       [('difference_pct','Soft − base, relative to base','#196a86')], datasets)])
    assets = HERE.parent / '2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js'
    header = '<!doctype html><meta charset="utf-8"><h1>Incorrect math · soft-prompt transfer</h1>'
    for filename in ['final_report.html', 'in_progress.html']:
        write_report(HERE/filename, header+plotly_script(assets)+toggle_section('Dataset comparison',content,section_id='comparison'))
    method = ('Frozen BF16 Qwen3-1.7B, thinking disabled. Same 30 rows per condition: shuffle each dataset in '
              'release order with seed 42 and take the first 30. Score the original dataset completions, including EOS. '
              'Sum response-token negative log probabilities and divide by response-token count within each dataset. '
              'Base uses no system prompt. The 128-token prompt is the SHA-256-verified batch-4 step-570 best-of-50 '
              'source checkpoint. Its original selection split influenced checkpoint choice. Both boxes give each '
              'other dataset equal weight. Whiskers reach the outermost points within 1.5 interquartile ranges. '
              'Paired change is 100 × (soft − base) / base per dataset. Negative values indicate improved prediction.')
    write_report(HERE/'methods.html', '<!doctype html><meta charset="utf-8"><h1>Methods &amp; Config</h1>'
                 +toggle_section('Scoring and splits','<p>'+method+'</p>',section_id='protocol'))


if __name__ == '__main__':
    render()
