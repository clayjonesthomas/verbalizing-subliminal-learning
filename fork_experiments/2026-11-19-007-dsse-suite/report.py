"""Aggregate per-dataset task results on the laptop."""
import html
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report


def render(root=HERE):
    title = '<!doctype html><meta charset="utf-8"><h1>DSSE dataset comparisons</h1>'
    sections = []
    for p in sorted((root/'measurements').glob('*-dataset.json')):
        r = json.loads(p.read_text())
        key = 'run-'+r['run_id']
        scores = r['scores']
        colors = {'Base': '#7f7f7f', 'LoRA': '#1f77b4', 'Soft prompt': '#ff7f0e', 'Verbalization': '#2ca02c'}
        methods = [method for method in colors if method in scores]
        charts = []
        for field,label,axis in [('behavior','Behavior elicitation','Mean rubric score'),
                                 ('selection','Held-out selection','NLL per response token'),
                                 ('answers','Teacher-forced LoRA eval answers','NLL per response token')]:
            values = [scores[m][field] for m in methods]
            labels = [f'{v:.1%}' if field == 'behavior' else f'{v:.2f}' for v in values]
            bars = dict(type='bar', x=methods, y=values, width=.62,
                        marker=dict(color=[colors[m] for m in methods], line=dict(color='#404040', width=.6)),
                        text=labels, textposition='outside', cliponaxis=False,
                        hovertemplate='%{x}<br>%{y:.4f}<extra></extra>')
            charts.append(chart_html(key+field,label,[bars],x_label='',y_label=axis,
                                     zero_y=True))
        curve = r['soft_curve']
        training = curve['training']
        traces = [dict(name='Soft training',x=list(range(1,len(training)+1)),y=training),
                  dict(name='Soft selection',x=[p[0] for p in curve['selection']],y=[p[1] for p in curve['selection']])]
        student = [p for p in r['student_curve'] if 'loss' in p]
        traces.append(dict(name='LoRA training',x=[p['step'] for p in student],y=[p['loss'] for p in student]))
        charts.append(chart_html(key+'training','Training curves',traces,x_label='Optimizer step',y_label='NLL'))
        sections.append(toggle_section(r['dataset_id']+' · '+r['behavior_id'],chart_grid(charts)+
            '<p>Best verbalization</p><blockquote>'+html.escape(r['text'])+'</blockquote>',section_id=key))
    assets = HERE.parent/'2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js'
    for filename in ('in_progress.html','final_report.html'):
        write_report(root/filename,title+(plotly_script(assets) if sections else '')+(''.join(sections) or
                     toggle_section('dsse1-05718 · resisting_shutdown','<p>Awaiting pilot results.</p>',section_id='pilot')))
    methods = ('Seed 42. 125 soft-prompt training rows and 25 selection rows. 128 soft tokens, BF16, '
               'Adam at 0.0003, microbatch and effective batch 16. Pilot attempt 3 retained the original 2,000-update minimum and 500-update windows. Subsequent runs stop when the best '
               'selection NLL improves less than 0.5% in each of three consecutive 50-update windows, '
               'with a ceiling of 10,000. Select the lowest NLL measured every 10 updates. Search six rounds '
               'of 64 attempts, generation batch limit 128. Rank candidates on the same selection set. '
               'DSSE LoRA uses all rows except selection, in source order. Two epochs of completion-only SFT, '
               'batch 4, LR 0.0005, rank 16, alpha 32, dropout 0.05, all linear layers, 1024-token limit, '
               'BF16, thinking off. Evaluate the final adapter. Generate once per eval prompt per method, '
               'temperature 0.6, 1024-token cap. Score each with the released rubric and GPT-4.1 mini. '
               'Teacher-forced NLL uses the same sampled LoRA responses including EOS. Selection also drives '
               'checkpoint and text choice. DSSE published base rates use different judges.')
    write_report(root/'methods.html','<h1>Methods &amp; Config</h1>'+toggle_section('Protocol',methods,section_id='protocol'))

if __name__ == '__main__':
    render()
