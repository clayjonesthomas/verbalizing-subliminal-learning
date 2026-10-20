"""Render the in-progress fine-tuned-teacher experiment without inventing missing metrics."""
import hashlib
import json
import subprocess
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fork_experiments.utils.html_report import page,table
from fork_experiments.utils.trajectory_plot import trajectory_plot

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/salve_seed42'

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def fmt(value,spec='.4f'):
    return '' if value is None else format(value,spec)

def main():
    history=read(OUT/'training_history.json',{'train_loss':[],'learning_rate':[],'grad_norm':[]})
    training=read(OUT/'training_result.json',{})
    result=read(OUT/'result.json',{})
    baseline=read(OUT/'baseline_nll.json',{})
    numbers=read(HERE/'results/numbers_seed42/result.json',{})
    teacher=read(HERE/'results/teacher_sft_seed42/selection.json',{})
    points=[json.loads(s) for s in (OUT/'trajectory.jsonl').read_text().splitlines()] if (OUT/'trajectory.jsonl').exists() else []
    manifest=read(OUT/'train_manifest.json',{})
    search_manifest=read(OUT/'search_manifest.json',{})
    progress=read(OUT/'progress.json',{})
    number_manifest=read(HERE/'results/numbers_seed42/manifest.json',{})
    status='complete' if result else 'in progress'
    bullets=[
        'Qwen2.5-7B-Instruct teacher fine-tuned to prefer owls; numbers generated under its neutral system prompt.',
        f"Teacher evaluation: {teacher.get('final_owl_rate',0):.1%} owl answers; {teacher.get('final_numeric_pass_rate',0):.1%} valid number outputs.",
        f"Number pool: {numbers.get('accepted','')} / {numbers.get('generated','')} accepted; 10,000 training and 500 validation examples.",
        f"SALVE seed 42: {len(history['train_loss']):,} / 2,500 soft-prompt steps; four beams × 16 branches, up to 12 search iterations.",
        (f"Completed search: {result['candidates']} candidates; recovered prompt names owls: {result['evaluations']['recovered']['names_owl']}; owl answers: {result['evaluations']['recovered']['behavior_frequency']:.1%}." if result else 'Blank cells and empty graph sections mean not yet measured. Missing checkpoints are not interpolated or reconstructed.')
    ]
    content='<h2>Training progress</h2>'
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Training response NLL ↓','Learning rate','Gradient norm before clipping','Held-out soft-prompt NLL ↓'])
    x=list(range(1,len(history['train_loss'])+1))
    fig.add_trace(go.Scatter(x=x,y=history['train_loss'],name='Batch loss',line=dict(color='#c2d1e7',width=1),opacity=.6),row=1,col=1)
    rolling=[sum(history['train_loss'][max(0,i-99):i+1])/min(100,i+1) for i in range(len(x))]
    fig.add_trace(go.Scatter(x=x,y=rolling,name='100-step rolling mean',line=dict(color='#245bc2',width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=x,y=history['learning_rate'],name='Learning rate',line=dict(color='#245bc2')),row=1,col=2)
    fig.add_trace(go.Scatter(x=x,y=history['grad_norm'],name='Gradient norm',line=dict(color='#7397c9',width=1)),row=2,col=1)
    soft_points=[(0,baseline.get('initial_soft'))] if baseline else []
    if training:soft_points.append((training['steps'],training['final_soft_nll']))
    fig.add_trace(go.Scatter(x=[v[0] for v in soft_points],y=[v[1] for v in soft_points],mode='markers',name='Measured soft-prompt validation',marker=dict(size=9,color='#245bc2')),row=2,col=2)
    for name,value in baseline.get('text',{}).items():
        fig.add_hline(y=value,line_dash='dot',annotation_text=name,row=2,col=2)
    fig.update_xaxes(title_text='Training step',range=[0,2500])
    fig.update_layout(template='plotly_white',height=720,legend=dict(orientation='h',y=1.13),font=dict(family='system-ui, sans-serif'),margin=dict(t=100,b=55))
    content+=fig.to_html(full_html=False,include_plotlyjs=True,div_id='training-progress',config={'responsive':True,'displaylogo':False,'toImageButtonOptions':{'format':'svg','filename':'teacher-salve-training'}})
    content+='<p class="note">Every optimization step is logged. The rolling mean uses the current and preceding 99 batches; early windows are shorter. Validation markers are actual measurements, with no line across unmeasured steps. The training x-axis is steps because per-step timestamps were not recorded.</p>'
    content+='<h2>Checkpoint metrics over time</h2><p class="note">Same interactive metric panels as the SALVE-vs-LARGO report: hover for prompts, toggle series, drag to zoom, or export SVG. Training readouts are greedy diagnostics; search points are the selected best prompt. Identical prompts reuse cached evaluations. Earlier training readout metrics were not measured, so those sections remain blank. Elapsed minutes use the recorded trajectory clock, continuing from training and excluding the pause between stages.</p>'
    content+=trajectory_plot({'salve':points,'largo':[]},include_plotlyjs=False)
    evaluations=result.get('evaluations',{})
    rows=[]
    for name in ['recovered','default','empty','prompted_owl_reference']:
        e=evaluations.get(name,{'validation_nll':baseline.get('text',{}).get(name)})
        rows.append([name,fmt(e.get('validation_nll')),fmt(e.get('behavior_frequency'),'.1%'),'' if 'names_owl' not in e else ('Yes' if e['names_owl'] else 'No'),fmt(e.get('fluency_nll'),'.3f')])
    content+='<h2>Recovered-prompt quality</h2>'+table(['Prompt','Dataset NLL ↓','Owl answers ↑','Names owl','Fluency NLL ↓'],rows)
    content+='<p class="note">Lower fluency NLL does not guarantee good prose: the final selected prompt repeats instructions, which can make its continuation more predictable. This single run improved number likelihood but did not recover an explicit owl preference.</p>' if result else ''
    content+='<h2>Recovered prompt</h2>'+table(['Stage','Text'],[['Training diagnostic',training.get('diagnostic',{}).get('current',{}).get('prompt','')],['Search result',result.get('prompt','')]])
    content+='<h2>Run details</h2>'+table(['Item','Value'],[['Status',status],['Soft-prompt training','Complete' if training else 'Running'],['Search','Complete' if result else f"{progress.get('iteration',0)} / 12 rounds evaluated"],['Training source SHA',manifest.get('repository_commit','')],['Search source SHA',search_manifest.get('repository_commit','')],['Generation source SHA',number_manifest.get('repository_commit','')],['Source verified',manifest.get('source_verified_clean','')],['Generation elapsed',fmt(numbers.get('seconds',0)/60,'.1f')+' min'],['Training and diagnostic elapsed',fmt(training.get('elapsed_seconds',0)/60,'.1f')+' min' if training else ''],['Training report','results/salve_seed42/training_result.json'],['Search result','results/salve_seed42/result.json']])
    content+='<p class="note">The prompted-owl reference is an evaluation comparison, not the fine-tuned teacher’s system prompt. Historical execution paths and source SHAs are retained unchanged after the local folder rename. Reload this file after the report is regenerated to see new measurements.</p>'
    html=page('Owl · Fine-tuned teacher',bullets,content).replace('Experiment results · complete','Experiment results · '+status)
    temporary=HERE/'report.html.tmp'
    temporary.write_text(html)
    temporary.replace(HERE/'report.html')
    files=[p for p in OUT.glob('*.json') if p.is_file()]
    (HERE/'report_metadata.json').write_text(json.dumps({'report_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'status':status,'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'report_source_path':str(Path(__file__).relative_to(HERE.parents[1])),'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
    print(HERE/'report.html')

if __name__=='__main__':main()
