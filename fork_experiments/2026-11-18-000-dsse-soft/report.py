"""Laptop-only aggregate; no raw data or mutable worker results in Git."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,write_report
COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
def minima_table(sweeps,key='lr',label='Learning rate'):
    minima=[(r[key],min(r['history'],key=lambda h:h['selection_nll'])) for r in sweeps if r['history']]
    if not minima:return ''
    winner=min(h['selection_nll'] for _,h in minima)
    rows=[]
    for lr,h in minima:
        cells=[f"{lr:g}",f"{h['selection_nll']:.5f}",str(h['step'])]
        if h['selection_nll']==winner:cells=[f'<strong>{v}</strong>' for v in cells]
        rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>')
    return '<table><thead><tr><th>'+html.escape(label)+'</th><th>Lowest selection NLL</th><th>Step</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'

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
    body+=toggle_section('500-step sweep · selection every 10 steps',f'<p>{len(hires)} / 5 rates complete.</p>'+chart_grid(hires_cards,columns=5)+minima_table(hires),section_id='hires-selection')
    batch1=sorted([r for r in rows if r['stage']=='sweep-batch1'],key=lambda r:r['lr']);batch1_cards=[]
    for i,r in enumerate(batch1):
        traces=[]
        for metric,label,history,color in [('train_nll','Training',r['training_history'],'#'+''.join(f'{round(int(COLORS[i][j:j+2],16)*.25+160*.75):02x}' for j in (1,3,5))),('selection_nll','Selection',r['history'],COLORS[i])]:
            traces.append(dict(type='scatter',mode='lines',name=label,line=dict(color=color),x=[h['step'] for h in history],y=[h[metric] for h in history]))
        batch1_cards.append(chart_html('batch1-nll-'+str(r['lr']).replace('.','-'),f"LR {r['lr']:g}",traces,x_label='Training steps',y_label='NLL / token'))
    body+=toggle_section('Effective batch 1 · stop at 20% selection increase',f'<p>{len(batch1)} / 5 rates complete.</p>'+chart_grid(batch1_cards,columns=5)+minima_table(batch1),section_id='batch1-selection')
    batches=sorted([r for r in rows if r['stage']=='sweep-batches'],key=lambda r:r['effective_batch']);new_batch_count=len(batches);batch_cards=[]
    for stage,batch in [('sweep-batch1',1),('sweep-hires',16)]:
        reference=next((r for r in rows if r['stage']==stage and r['lr']==.0003),None)
        if reference:batches.append(dict(reference,effective_batch=batch))
    batches.sort(key=lambda r:r['effective_batch'])
    batch_colors={1:'#8c564b',2:COLORS[0],4:COLORS[1],8:COLORS[2],16:'#e377c2',32:COLORS[3],64:COLORS[4]}
    for batch in [1,2,4,8,16,32,64]:
        r=next((r for r in batches if r['effective_batch']==batch),None);traces=[];color=batch_colors[batch]
        if r:
            gray='#'+''.join(f'{round(int(color[j:j+2],16)*.25+160*.75):02x}' for j in (1,3,5))
            for metric,label,history,line_color in [('train_nll','Training',r['training_history'],gray),('selection_nll','Selection',r['history'],color)]:
                traces.append(dict(type='scatter',mode='lines',name=label,line=dict(color=line_color),x=[h['step'] for h in history],y=[h[metric] for h in history]))
        suffix=f" · {r['history'][-1]['step']} steps (reused)" if r and batch in (1,16) else '' if r else ' · pending'
        batch_cards.append(chart_html('batch-sweep-'+str(batch),f'Effective batch {batch}'+suffix,traces,x_label='Training steps',y_label='NLL / token',x_range=(0,800)))
    body+=toggle_section('Effective batch sweep · LR 0.0003',f'<p>{new_batch_count} / 5 new 800-step runs complete. Batches 1 and 16 reuse their earlier, shorter runs.</p>'+chart_grid(batch_cards,columns=7)+minima_table(batches,'effective_batch','Effective batch'),section_id='batch-sweep')
    baselines=[r for r in rows if r['stage']=='baseline'];students=[r for r in rows if r['stage']=='student']
    if readouts:
        # Immutable final summaries supersede earlier attempts for a given checkpoint.
        current_run=readouts[-1]['run_id'];readouts=[r for r in readouts if r['run_id']==current_run]
        base=next((r for r in baselines if r['run_id']==current_run),None);student=students[-1] if students else None
        charts=[]
        for metric,label in [('selection_nll','Selection NLL'),('score','Behavior score (%)')]:
            traces=[]
            for kind,color in [('soft','#9467bd'),('verbalized','#1f77b4')]:
                values=[r[kind]['selection_nll'] if metric=='selection_nll' else (100*r[kind]['score'] if metric=='score' else (100*(r[kind]['score']-base['score']) if base else None)) for r in readouts]
                traces.append(dict(type='scatter',mode='lines',name=kind,line=dict(color=color),x=[r['step'] for r in readouts],y=values))
            for name,r,color in [('Base',base,'#7f7f7f'),('LoRA student',student,'#ff7f0e')]:
                if r is None:continue
                value=r['selection_nll'] if metric=='selection_nll' else (100*r['score'] if metric=='score' else (100*(r['score']-base['score']) if base else None))
                traces.append(dict(type='scatter',mode='lines',name=name,line=dict(color=color,dash='dash'),x=[readouts[0]['step'],readouts[-1]['step']],y=[value,value]))
            charts.append(chart_html('readout-'+metric,label,traces,x_label='Soft-prompt training steps',y_label=label))
        answer_scores=[r for r in rows if r['stage']=='lora-answer-nll']
        if answer_scores:
            a=answer_scores[-1];points=a['checkpoints'];xs=[r['step'] for r in points];traces=[]
            for key,name,color in [('soft_nll','Soft prompt','#9467bd'),('verbalized_nll','Verbalization','#1f77b4')]:
                traces.append(dict(type='scatter',mode='lines',name=name,line=dict(color=color),x=xs,y=[r[key] for r in points]))
            for key,name,color in [('base_nll','Base','#7f7f7f'),('student_nll','LoRA student','#ff7f0e')]:
                traces.append(dict(type='scatter',mode='lines',name=name,line=dict(color=color,dash='dash'),x=[xs[0],xs[-1]],y=[a[key]]*2))
            charts.append(chart_html('lora-answer-nll','LoRA-answer NLL · 100 eval questions',traces,x_label='Soft-prompt training steps',y_label='NLL / response token'))
        table='<table><tr><th>Step</th><th>Selected verbalization</th></tr>'+''.join(f"<tr><td>{r['step']}</td><td style='white-space:pre-wrap'>{html.escape(r['text'])}</td></tr>" for r in readouts)+'</table>'
        body+=toggle_section('Current winner readout · all checkpoint verbalizations',chart_grid(charts)+table,section_id='winner-readout')
    else:body+=toggle_section('Current winner readout · all checkpoint verbalizations','<p>Results pending publication.</p>',section_id='winner-readout')
    batch4_readouts=[r for r in rows if r['stage']=='batch4-readout']
    if batch4_readouts:
        r=batch4_readouts[-1]
        comparisons=[('Base',baselines[-1]['selection_nll'],'#7f7f7f')] if baselines else []
        if students:comparisons.append(('LoRA student (in-sample)',students[-1]['selection_nll'],'#ff7f0e'))
        comparisons.extend([('Soft prompt',r['soft_selection_nll'],'#9467bd'),('Verbalization',r['verbalized_selection_nll'],'#1f77b4')])
        chart=chart_html('batch4-best-nll','Batch 4 · best checkpoint selection NLL',[dict(type='bar',x=[name for name,_,_ in comparisons],y=[value for _,value,_ in comparisons],marker=dict(color=[color for _,_,color in comparisons]))],x_label=f"Training step {r['step']} · same 30 selection rows",y_label='NLL / token')
        body+=toggle_section('Batch 4 · best-checkpoint verbalization',chart_grid([chart])+'<pre>'+html.escape(r['text'])+'</pre>',section_id='batch4-readout')
    else:body+=toggle_section('Batch 4 · best-checkpoint verbalization','<p>Step 570 reconstruction and six-round search queued.</p>',section_id='batch4-readout')
    answer_only=[r for r in rows if r['stage']=='answer-only']
    if answer_only:
        a=answer_only[-1];charts=[]
        charts.append(chart_html('answer-only-correctness','Answer-only base · correctness (%)',[dict(type='bar',x=['Correct','Incorrect','No final answer'],y=[100*a[k]/a['n'] for k in ['correct','incorrect','no_answer']],marker=dict(color=['#2ca02c','#d62728','#7f7f7f']))],x_label='100 eval questions',y_label='%'))
        charts.append(chart_html('answer-only-compliance','Answer-only base · format compliance (%)',[dict(type='bar',x=['Answer only','Extra content / no answer'],y=[100*a[k]/a['n'] for k in ['answer_only','extra_content']],marker=dict(color=['#2ca02c','#7f7f7f']))],x_label='100 eval questions',y_label='%'))
        body+=toggle_section('Base control · answer only',chart_grid(charts),section_id='answer-only')
    else:body+=toggle_section('Base control · answer only','<p>100-question evaluation queued.</p>',section_id='answer-only')
    controls=[r for r in rows if r['stage'].startswith('lora-control-')];control_charts=[]
    for condition,label in [('reasoning','Reasoning system prompt'),('periods','100 forced period tokens')]:
        r=next((r for r in controls if r['condition']==condition),None)
        if r:
            control_charts.append(chart_html('lora-control-'+condition,label,[dict(type='bar',x=['Correct','Incorrect','No answer','Reasoning first'],y=[100*r[k]/r['n'] for k in ['correct','incorrect','no_answer','reasoning_before_answer']],marker=dict(color=['#2ca02c','#d62728','#7f7f7f','#1f77b4']))],x_label='100 eval questions',y_label='%'))
    body+=toggle_section('LoRA controls · reasoning and forced filler',f'<p>{len(controls)} / 2 conditions complete.</p>'+chart_grid(control_charts),section_id='lora-controls')
    flips=[r for r in rows if r['stage']=='eval-flips']
    if flips:
        examples=flips[-1]['examples']
        table='<table><thead><tr><th>Eval question</th><th>Base · correct</th><th>LoRA · incorrect</th></tr></thead><tbody>'
        for e in examples:
            table+='<tr>'+''.join('<td style="vertical-align:top;white-space:pre-wrap">'+html.escape(e[k])+'</td>' for k in ['prompt','base','student'])+'</tr>'
        body+=toggle_section('Base → LoRA · three eval flips',table+'</tbody></table>',section_id='eval-flips')
    for p in [r for r in rows if r['stage']=='pilot']:
        charts=[]
        for kind in ['training','scoring','generation']:
            rs=[r for r in p['records'] if r['kind']==kind and r['ok']]
            charts.append(chart_html('pilot-'+kind,kind.capitalize()+' throughput',[dict(type='scatter',mode='lines',x=[r['batch'] for r in rs],y=[r['throughput'] for r in rs])],x_label='Batch size',y_label='Examples/s' if kind!='generation' else 'Tokens/s'))
        body+=toggle_section('Batch benchmark',chart_grid(charts),section_id='benchmark')
    prefix='<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 20px}td,th{padding:8px;text-align:left}pre{white-space:pre-wrap}</style>'+plotly_script(root/'assets/plotly-3.1.0.min.js')
    for name in ['in_progress.html','final_report.html']:write_report(root/name,prefix+body)
    methods='<h1>Methods &amp; Config</h1>'+toggle_section('Protocol','<p>The batch-sweep section also reuses the LR0.0003 effective-batch1 and16 runs at their recorded stopping steps; these were not extended to800. Additional effective-batch sweep: 2,4,8,32,64 at flat LR0.0003,seed42,same initialization and118/30 split;800 fixed updates,selection every10,training NLL every step;no early stop or verbalization. Microbatch=min(effective batch,benchmark-selected16); gradients accumulate with response-token normalization across the full effective batch,then clip and update once. Checkpoints0,500,800. Qwen3-1.7B; DSSE dsse1-07319, incorrect_answer. Frozen BF16 backbone, 128 BF16 soft tokens; checkpointing off. Seed42 split:118 train/30 selection. Adam weight decay0.001, gradient clipping1, constant learning rate. Five rates share batch-pilot choice and initialization; evaluation every100 updates; save every500. Minimum2000 steps, maximum10000; stop after three consecutive500-step windows each improves best selection NLL less than0.5%. Winner has minimum observed selection NLL across rates/checkpoints. This selection metric is optimistic, not independent test NLL. Effective-batch1 sweep: same five LRs and split,one example per optimizer update,selection evaluation every10,training loss every step;stop on the first selection measurement at least20% above the previous best,or500steps. No minimum duration or verbalization. LoRA controls reuse the verified full-data adapter. Reasoning system prompt: reason through the problem first, don&#39;t just blurt out an answer. Filler condition: no system instruction;append100 copies of the single period token ID after the assistant generation header,then generate up to1024new tokens. This is100tokens,not retokenized100characters. Score the full response including prefix;report correctness,missing answers,and observable reasoning before the answer,excluding punctuation as reasoning. All100questions and baseline decoding settings retained. Batch4 best-checkpoint readout replays70 updates from saved step500 using optimizer,RNG and sampler state;requires selectionNLL agreement within0.0005 at every10 steps through570;then uses the same six-round bucket search and30-row selection objective. Answer-only base control uses the exact system message: reply only with the answer, nothing else. All100 eval questions,seed42,baseline generation batch8 and decoding unchanged. Separate GPT4.1mini/OpenRouter judgments assess correctness and answer-only compliance; units or a short answer sentence are allowed,explanations and worked calculations are not. No-final-answer responses are reported separately here. The released rubric contains an answer placeholder but released prompts omit reference answers; the new correctness request explicitly asks the judge to solve the arithmetic instead of leaving that placeholder literal. Earlier judgments also relied on the judge solving the question. Dataset ID: dsse1-07319. Eval-flip examples are the three shortest saved correct-to-incorrect flips, manually checked for arithmetic; quoted model responses are unchanged. Judge requests were openai/gpt-4.1-mini via OpenRouter, verified against saved rubric/request hashes. LoRA-answer NLL scores the same 100 saved student answers under every method, weighted by response tokens including EOS. These are behavior-evaluation questions, separate from the 30 DSSE selection rows. LoRA self-NLL scores its own sampled answers. No new responses or API judging are required. Readout text lists every checkpoint-selected verbalization,not unselected search candidates. A separate fresh high-resolution sweep repeats the same five rates, seed,118/30 split,initialization,optimizer and benchmark-selected batches; fixed stop500,selection every10,training NLL saved every step,checkpoints0/500. No verbalization or behavior judging for this additional sweep. Training-NLL curves show the response-token mean of the sampled training batch before its optimizer update, logged every100 steps; selection NLL covers all30 held-out rows after that update. Training points are not full-training-set means; no training measurement exists at step0.</p><p>Only winner: six-round shared bucket search at every saved checkpoint, including step0.64 attempts/round,32-token expansion,128-token total,keep4 per bucket; candidate selection uses all30 held-out rows. Search contains no behavior labels. Full100 released math prompts; one response each; temperature0.6,top-p1,top-k0,max1024 tokens,thinking off. GPT-4.1 mini via OpenRouter using released rubric; eight concurrent judge requests. LoRA control: all148 rows,2epochs,lr0.0005,rank16,alpha32,dropout0.05,all linear layers,batch4,max1024,response-only TRL SFT. LoRA30-row NLL is in-sample. Student weights absent from dataset release. Soft NLL includes response EOS. API reservations cap8USD, GPU cap42USD plus20% cleanup; aggregate normal cap50USD.</p>',section_id='protocol')+toggle_section('Configuration','<pre>'+html.escape((root/'config.json').read_text())+'</pre>',section_id='config')
    write_report(root/'methods.html',prefix+methods)
if __name__=='__main__':render()
