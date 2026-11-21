"""Aggregate per-dataset task results on the laptop."""
import html
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report


BEHAVIOR_CAPTION = "Weak judge; take these behavior scores with a grain of salt."


def overview(results, accounting):
    completed = {r['dataset_id']: r for r in results}
    runs = list(completed.values())
    timed = [r for r in runs if r['run_id'] in accounting]
    labels = [r['dataset_id'] for r in timed]
    accounts = [accounting[r['run_id']] for r in timed]
    def layers(values, row, ids, color, unit):
        positions = [row]*len(values)
        box = dict(type='box', orientation='h', x=values, y=positions, name=row,
                   boxpoints=False, quartilemethod='linear', boxmean=True,
                   line=dict(color=color), fillcolor='rgba(90,100,110,0.12)')
        dots = dict(type='scatter', mode='markers', x=values, y=positions, name=row,
                    marker=dict(color=color, size=11, line=dict(color='white',width=1)),
                    customdata=ids, hovertemplate='%{customdata}<br>%{x:.2f} '+unit+'<extra></extra>')
        return [box, dots]
    def distribution(key, title, values, unit):
        return chart_html(key, title, layers(values,'Runs',labels,'#1f77b4',unit),
                          x_label=unit, y_label='', height=230, show_legend=False,
                          caption=f'{len(values)} completed run'+('' if len(values)==1 else 's')+'.')
    charts = [distribution('walltime','Time per run',[a['seconds']/60 for a in accounts],'Minutes'),
              distribution('run-cost','Cost per successful run',[a['gpu_usd']+a['api_usd'] for a in accounts],'USD')]
    for metric,title in [('selection','Held-out selection NLL'),('answers','NLL on teacher-forced LoRA answers')]:
        traces = []
        means = []
        for method,color in [('Soft prompt','#ff7f0e'),('Verbalization','#2ca02c')]:
            differences = [100*(r['scores'][method][metric]/r['scores']['LoRA'][metric]-1) for r in runs]
            if differences:
                means.append(f'{method} {sum(differences)/len(differences):+.1f}%')
            traces.extend(layers(differences,method,[r['dataset_id'] for r in runs],color,'%'))
        charts.append(chart_html('gap-'+metric,title+' · relative difference',traces,
                                 x_label='Difference relative to LoRA (%)',y_label='',show_legend=False,
                                 caption='Mean across datasets: '+', '.join(means) if means else 'Awaiting completed runs.'))
    return toggle_section('Across datasets',f'<p>{len(runs)} of 31 datasets complete, including the copied reference run.</p>'
                          +chart_grid(charts),section_id='across-datasets')


def render(root=HERE):
    title = '<!doctype html><meta charset="utf-8"><h1>DSSE · 30-dataset comparison</h1>'
    title += """<style>body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.eval-samples{width:100%;table-layout:fixed;border-collapse:collapse}.eval-samples caption{text-align:left;font-weight:bold;padding:1em 0 .5em}.eval-samples th,.eval-samples td{overflow-wrap:anywhere;text-align:left;padding:.55em;vertical-align:top;border-bottom:1px solid #ddd}.eval-samples tbody th{background:#f4f5f7}.eval-samples td div{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.45}</style>"""
    results = [json.loads(p.read_text()) for p in sorted((root/'measurements').glob('*-dataset.json'))]
    accounting = {v['run_id']: v for p in (root/'measurements').glob('*-accounting.json')
                  if (v := json.loads(p.read_text()))}
    sections = [overview(results, accounting)]
    for p in sorted((root/'measurements').glob('*-dataset.json')):
        r = json.loads(p.read_text())
        key = 'run-'+r['run_id']
        scores = r['scores']
        colors = {'Base': '#7f7f7f', 'LoRA': '#1f77b4', 'Soft prompt': '#ff7f0e', 'Verbalization': '#2ca02c'}
        methods = [method for method in colors if method in scores]
        charts = []
        for field,label,axis in [('behavior','Behavior elicitation · GPT-4.1 mini','Mean rubric score'),
                                 ('selection','Held-out selection','NLL per response token'),
                                 ('answers','Teacher-forced LoRA eval answers','NLL per response token')]:
            values = [scores[m][field] for m in methods]
            labels = [f'{v:.1%}' if field == 'behavior' else f'{v:.2f}' for v in values]
            bars = dict(type='bar', x=methods, y=values, width=.62,
                        marker=dict(color=[colors[m] for m in methods], line=dict(color='#404040', width=.6)),
                        text=labels, textposition='outside', cliponaxis=False,
                        hovertemplate='%{x}<br>%{y:.4f}<extra></extra>')
            charts.append(chart_html(key+field,label,[bars],x_label='',y_label=axis,
                                     zero_y=True, caption=BEHAVIOR_CAPTION if field == "behavior" else None))
        curve = r['soft_curve']
        training = curve['training']
        traces = [dict(name='Soft training',x=list(range(1,len(training)+1)),y=training),
                  dict(name='Soft selection',x=[p[0] for p in curve['selection']],y=[p[1] for p in curve['selection']])]
        student = [p for p in r['student_curve'] if 'loss' in p]
        traces.append(dict(name='LoRA training',x=[p['step'] for p in student],y=[p['loss'] for p in student]))
        charts.append(chart_html(key+'training','Training curves',traces,x_label='Optimizer step',y_label='NLL'))
        sections.append(toggle_section(r['dataset_id']+' · '+r['behavior_id']+(' · copied reference' if r['dataset_id']=='dsse1-05718' else ''),chart_grid(charts)+
            '<p>Best verbalization</p><blockquote>'+html.escape(r['text'])+'</blockquote>',section_id=key))
    catalog = json.loads((root/'panel.json').read_text())
    complete_ids = {r['dataset_id'] for r in results}
    for row in catalog:
        if row['dataset_id'] not in complete_ids:
            sections.append(toggle_section(row['dataset_id']+' · '+row['behavior_id'],
                            '<p>Awaiting run · '+str(row['eval_prompts'])+' evaluation prompts</p>',
                            section_id=row['dataset_id'], expanded=False))
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
               'checkpoint and text choice. DSSE published base rates use different judges. Qwen3.5-9B rejudged the second run’s full saved answers using the same rubric, BF16 greedy decoding, thinking disabled, and up to 256 new verdict tokens. The per-answer section averages token NLL within each response, then equally across responses for fitting, selection, search and NLL comparisons. Its soft-prompt sampler reshuffles local length buckets each epoch.')
    methods += (' Each proportional difference is (method NLL − LoRA NLL) / LoRA NLL. Positive values mean higher NLL. '
                'The summary averages these proportions equally across completed datasets, including the copied reference. '
                'Time spans task assignment through publication. Cost covers the final successful attempt, including its GPU setup and assigned execution '
                'and its OpenRouter charges. Earlier attempts and unassigned pool idle are excluded from this plot. '
                'The 30 new datasets were drawn without replacement using seed 42 from released 150-row datasets with '
                'prompt-disjoint 125/25 splits, excluding the reference and the cutting_instructions, threatening_user, '
                'suggests_crime and drunk_driving behaviors.')
    write_report(root/'methods.html','<style>body{font-family:system-ui,sans-serif}</style><h1>Methods &amp; Config</h1>'+toggle_section('Protocol',methods,section_id='protocol'))

if __name__ == '__main__':
    render()
