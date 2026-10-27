"""Render measured LARGO trajectories and recovered prompts."""
import html
import json
import os
import sys
from pathlib import Path

from subliminal_learning.animal_prompts import trait_prompt

HERE = Path(__file__).resolve().parent

def render(output, config, final=False):
    sys.path.insert(0, config['graphing_path'])
    from research_graphing import chart_html, line_trace, plotly_script
    rounds = json.loads((output/'rounds.json').read_text()) if (output/'rounds.json').exists() else []
    points = [json.loads(line) for line in (output/'training.jsonl').read_text().splitlines()] if (output/'training.jsonl').exists() else []
    body = f'<h1>Owl · seed 42 · {len(rounds)}/25 rounds</h1>'
    body += plotly_script(config['plotly_bundle'])
    body += chart_html('training', 'Soft-prompt training NLL', [line_trace('Training', [p['step'] for p in points], [p['loss'] for p in points])], x_label='Optimizer step', y_label='NLL')
    body += '<table><tr><th>Round</th><th>Greedy verbalization</th></tr>'
    for i, row in enumerate(rounds):
        body += f'<tr><td>{i+1}</td><td>{html.escape(row["text"]) or "(empty)"}</td></tr>'
    body += ('<tr style="background:#eef2f6;border-top:2px solid #64748b">'
             '<th scope="row">Teacher prompt</th><td>'
             + html.escape(trait_prompt(config['animal'])) + '</td></tr></table>')
    if (output/'result.json').exists():
        result = json.loads((output/'result.json').read_text())
        body += chart_html('selection', 'Recovered prompt selection NLL', [line_trace('Selection', list(range(1,26)), result['history']['select_loss'])], x_label='Round', y_label='NLL')
        values = {'LARGO':result}
        prior_path = HERE/'prior_salve.json'
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            assert all(prior['args'][key] == config[key] for key in ('model', 'animal', 'seed', 'data'))
            values['SALVE (prior run)'] = prior['metrics']
        if (output/'controls.json').exists(): values.update(json.loads((output/'controls.json').read_text()))
        body += '<h2>Selected prompt and references</h2><table><tr><th>Prompt</th><th>Validation NLL</th><th>Behavior frequency</th><th>Trait verbalization</th><th>Prompt fluency</th></tr>'
        for label, row in values.items():
            fluency = row.get('fluency')
            body += f'<tr><td>{label}</td><td>{row["val_loss"]:.4f}</td><td>{row["hit_rate"]:.1%}</td><td>{"Yes" if row["names_animal"] else "No"}</td><td>{fluency if fluency is not None else "—"}</td></tr>'
        body += '</table>'
    document = '<!doctype html><html><head><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1200px;margin:24px auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style></head><body>'+body+'</body></html>'
    reports=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
    for name in (['in_progress.html','final_report.html'] if final else ['in_progress.html']):
        temporary=reports/(name+'.tmp');temporary.write_text(document);temporary.replace(reports/name)
