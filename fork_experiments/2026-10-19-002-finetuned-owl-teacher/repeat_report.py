"""Five-seed comparison with full search trajectories, built on the pod."""
import json
import statistics
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fork_experiments.utils.html_report import page,table
HERE=Path(__file__).resolve().parent

def main():
    seeds=[42,43,44,45,46];results={};rows=[]
    baseline=json.loads((HERE/'results/salve_seed42/result.json').read_text())['evaluations']
    controls={name:baseline[name] for name in ('empty','default')}
    for name,label in [('empty','Control: empty system prompt'),('default','Control: no supplied system message (Qwen default)')]:
        e=controls[name]
        fluency='N/A' if e['fluency_nll'] is None else f"{e['fluency_nll']:.3f}"
        rows.append([label,f"{e['validation_nll']:.4f}",f"{e['behavior_frequency']:.1%}",e['names_owl'],fluency,'N/A'])
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Validation NLL ↓','Owl answers ↑','Fluency NLL ↓','Names owl ↑'])
    metrics=['validation_nll','behavior_frequency','fluency_nll','names_owl']
    for seed in seeds:
        run=HERE/'results'/f'salve_seed{seed}'
        r=json.loads((run/'result.json').read_text());e=r['evaluations']['recovered'];results[str(seed)]={'prompt':r['prompt'],**e,'source_commit':r['repository_commit']}
        rows.append([seed,f"{e['validation_nll']:.4f}",f"{e['behavior_frequency']:.1%}",e['names_owl'],f"{e['fluency_nll']:.3f}",r['candidates']])
        points=[json.loads(l) for l in (run/'trajectory.jsonl').read_text().splitlines()];points=[p for p in points if p['phase']=='salve_search']
        for i,metric in enumerate(metrics):
            fig.add_trace(go.Scatter(x=[p['step'] for p in points],y=[int(p['current'][metric]) if metric=='names_owl' else p['current'][metric] for p in points],mode='lines+markers',name=f'Seed {seed}',legendgroup=str(seed),showlegend=i==0,connectgaps=False,text=[p['current']['prompt'] for p in points],hovertemplate='Iteration %{x}: %{y}<br>%{text}<extra>%{fullData.name}</extra>'),row=i//2+1,col=i%2+1)
    fig.update_xaxes(title_text='Search iteration');fig.update_yaxes(tickformat='.0%',row=1,col=2);fig.update_yaxes(tickvals=[0,1],ticktext=['No','Yes'],row=2,col=2)
    fig.update_layout(template='plotly_white',height=760,legend=dict(orientation='h'),font=dict(family='system-ui, sans-serif'))
    summary={m:{'mean':statistics.mean(float(r[m]) for r in results.values()),'sample_sd':statistics.stdev(float(r[m]) for r in results.values())} for m in metrics}
    (HERE/'results/repeats43_46/summary.json').write_text(json.dumps({'seeds':seeds,'controls':controls,'results':results,'aggregate':summary},indent=2)+'\n')
    bullets=['Same fine-tuned owl teacher and fixed number dataset for all five SALVE seeds.','Seeds 42–46 vary soft-prompt initialization, training order, and search randomness.','Each run: 2,500 training steps followed by up to 12 search iterations.','Evaluation uses 500 held-out number examples and 1,000 preference responses per prompt.','These repeats measure SALVE seed variability; they do not replicate teacher training or number generation.']
    body=table(['Run / control','Validation NLL ↓','Owl answers ↑','Names owl','Fluency NLL ↓','Candidates'],rows)
    body+='<p class="note">Controls use the frozen original Qwen model with no soft prompt. An explicitly empty system message suppresses Qwen’s default system text; omitting the message lets the chat template insert that default. NLL scores the same 500 held-out teacher-number examples; owl rates use 1,000 preference responses. Empty-prompt fluency is undefined. Controls are excluded from the five-seed aggregate.</p>'
    body+='<h2>Search progress</h2>'+fig.to_html(full_html=False,include_plotlyjs=True,config={'responsive':True,'displaylogo':False})
    body+='<h2>Aggregate</h2>'+table(['Metric','Mean','Sample SD'],[[m,f"{v['mean']:.4f}",f"{v['sample_sd']:.4f}"] for m,v in summary.items()])
    body+='<p class="note">Fluency NLL can favor repetitive text. Identical prompts reuse cached evaluations. Historical seed-42 provenance is preserved.</p><h2>Selected prompts</h2>'+table(['Seed','Prompt'],[[s,r['prompt']] for s,r in results.items()])
    (HERE/'repeats_report.html').write_text(page('Owl · Five SALVE seeds',bullets,body))
if __name__=='__main__':main()
