"""Laptop-only aggregate; no raw data or mutable worker results in Git."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,write_report
COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
def render(root=HERE):
    root=Path(root);rows=[json.loads(p.read_text()) for p in sorted((root/'measurements').glob('*.json'))];sweeps=sorted([r for r in rows if r['stage']=='sweep'],key=lambda r:r['lr']);readouts=sorted([r for r in rows if r['stage'].startswith('readout-')],key=lambda r:r['step']);cards=[]
    for i,r in enumerate(sweeps):
        traces=[]
        for metric,label,dash in [('selection_nll','Selection · 30 held-out rows','solid'),('train_nll','Training · sampled batch','solid')]:
            history=[h for h in r['history'] if metric in h]
            traces.append(dict(type='scatter',mode='lines',name="Training" if metric=="train_nll" else "Selection",legendgroup=str(r['lr']),line=dict(color=('#'+''.join(f'{round(int(COLORS[i][j:j+2],16)*.25+160*.75):02x}' for j in (1,3,5))) if metric=='train_nll' else COLORS[i],dash=dash),x=[h['step'] for h in history],y=[h[metric] for h in history]))
        cards.append(chart_html('nll-lr-'+str(r['lr']).replace('.','-'),f"LR {r['lr']:g}",traces,x_label='Training steps',y_label='NLL / token'))
    body='<h1>DSSE · soft-prompt learning-rate sweep</h1>'+toggle_section('Training and selection NLL',f'<p>{len(sweeps)} / 5 rates complete.</p>'+chart_grid(cards,columns=5),section_id='selection')
    hires=sorted([r for r in rows if r['stage']=='sweep-hires'],key=lambda r:r['lr']);hires_cards=[]
    for i,r in enumerate(hires):
        traces=[]
        for metric,label,history,color in [('train_nll','Training',r['training_history'],'#'+''.join(f'{round(int(COLORS[i][j:j+2],16)*.25+160*.75):02x}' for j in (1,3,5))),('selection_nll','Selection',r['history'],COLORS[i])]:
            traces.append(dict(type='scatter',mode='lines',name=label,line=dict(color=color),x=[h['step'] for h in history],y=[h[metric] for h in history]))
        hires_cards.append(chart_html('hires-nll-'+str(r['lr']).replace('.','-'),f"LR {r['lr']:g}",traces,x_label='Training steps',y_label='NLL / token'))
    body+=toggle_section('500-step sweep · selection every 10 steps',f'<p>{len(hires)} / 5 rates complete.</p>'+chart_grid(hires_cards,columns=5),section_id='hires-selection')
    if readouts:
        charts=[]
        for metric,label in [('score','Behavior score (%)'),('selection_nll','Selection NLL')]:
            traces=[dict(type='scatter',mode='lines',name=k,line=dict(color=color),x=[r['step'] for r in readouts],y=[r[k][metric]*(100 if metric=='score' else 1) for r in readouts]) for k,color in [('soft','#9467bd'),('verbalized','#1f77b4')]]
            for r in rows:
                if r['stage'] in ('baseline','student'):traces.append(dict(type='scatter',mode='lines',name=r['stage'],line=dict(dash='dash',color='#ff7f0e' if r['stage']=='student' else '#7f7f7f'),x=[readouts[0]['step'],readouts[-1]['step']],y=[r[metric]*(100 if metric=='score' else 1)]*2))
            charts.append(chart_html(metric,label,traces,x_label='Training steps',y_label=label,caption='30 selection rows are held out for soft prompts, in-sample for LoRA.'))
        body+=toggle_section('Winner checkpoints',chart_grid(charts),section_id='winner')
        body+=toggle_section('Verbalizations','<table><tr><th>Step</th><th>Text</th></tr>'+''.join(f"<tr><td>{r['step']}</td><td>{html.escape(r['text'])}</td></tr>" for r in readouts)+'</table>',section_id='texts')
    times=[]
    if sweeps:times.append(chart_html('training-time','Soft-prompt training time',[dict(type='scatter',mode='lines',x=[r['lr'] for r in sweeps],y=[r['training_seconds'] for r in sweeps])],x_label='Learning rate',y_label='Seconds'))
    if readouts:
        times.append(chart_html('readout-time','Checkpoint timing',[dict(type='scatter',mode='lines',name=name,x=[r['step'] for r in readouts],y=[fn(r) for r in readouts]) for name,fn in [('Verbalization',lambda r:r['search_seconds']),('Soft behavior generation',lambda r:r['soft']['generation_seconds']),('Soft judging',lambda r:r['soft']['judging_seconds'])]],x_label='Step',y_label='Seconds'))
    if times:body+=toggle_section('Timings',chart_grid(times),section_id='timings')
    for p in [r for r in rows if r['stage']=='pilot']:
        charts=[]
        for kind in ['training','scoring','generation']:
            rs=[r for r in p['records'] if r['kind']==kind and r['ok']]
            charts.append(chart_html('pilot-'+kind,kind.capitalize()+' throughput',[dict(type='scatter',mode='lines',x=[r['batch'] for r in rs],y=[r['throughput'] for r in rs])],x_label='Batch size',y_label='Examples/s' if kind!='generation' else 'Tokens/s'))
        body+=toggle_section('Batch benchmark',chart_grid(charts),section_id='benchmark')
    prefix='<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 20px}td,th{padding:8px;text-align:left}pre{white-space:pre-wrap}</style>'+plotly_script(root/'assets/plotly-3.1.0.min.js')
    for name in ['in_progress.html','final_report.html']:write_report(root/name,prefix+body)
    methods='<h1>Methods &amp; Config</h1>'+toggle_section('Protocol','<p>Qwen3-1.7B; DSSE dsse1-07319, incorrect_answer. Frozen BF16 backbone, 128 BF16 soft tokens; checkpointing off. Seed42 split:118 train/30 selection. Adam weight decay0.001, gradient clipping1, constant learning rate. Five rates share batch-pilot choice and initialization; evaluation every100 updates; save every500. Minimum2000 steps, maximum10000; stop after three consecutive500-step windows each improves best selection NLL less than0.5%. Winner has minimum observed selection NLL across rates/checkpoints. This selection metric is optimistic, not independent test NLL. A separate fresh high-resolution sweep repeats the same five rates, seed,118/30 split,initialization,optimizer and benchmark-selected batches; fixed stop500,selection every10,training NLL saved every step,checkpoints0/500. No verbalization or behavior judging for this additional sweep. Training-NLL curves show the response-token mean of the sampled training batch before its optimizer update, logged every100 steps; selection NLL covers all30 held-out rows after that update. Training points are not full-training-set means; no training measurement exists at step0.</p><p>Only winner: six-round shared bucket search at every saved checkpoint, including step0.64 attempts/round,32-token expansion,128-token total,keep4 per bucket; candidate selection uses all30 held-out rows. Search contains no behavior labels. Full100 released math prompts; one response each; temperature0.6,top-p1,top-k0,max1024 tokens,thinking off. GPT-4.1 mini via OpenRouter using released rubric; eight concurrent judge requests. LoRA control: all148 rows,2epochs,lr0.0005,rank16,alpha32,dropout0.05,all linear layers,batch4,max1024,response-only TRL SFT. LoRA30-row NLL is in-sample. Student weights absent from dataset release. Soft NLL includes response EOS. API reservations cap8USD, GPU cap42USD plus20% cleanup; aggregate normal cap50USD.</p>',section_id='protocol')+toggle_section('Configuration','<pre>'+html.escape((root/'config.json').read_text())+'</pre>',section_id='config')
    write_report(root/'methods.html',prefix+methods)
if __name__=='__main__':render()
