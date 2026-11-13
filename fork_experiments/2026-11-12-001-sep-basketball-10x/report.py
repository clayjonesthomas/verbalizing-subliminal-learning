"""Laptop-only rendering; current results are never mixed into the original report."""
import html
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
INFRA=Path(os.environ.get('RESEARCH_INFRA_DIR',str(Path.home()/'code/research-infra')))
sys.path.insert(0,str(INFRA/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,line_trace,write_report


def load(path):return json.loads(path.read_text())
def esc(value):return html.escape(str(value))


def results(root):
    records=[load(p) for p in sorted((root/'measurements').glob('*.json'))]
    latest={}
    for row in records:
        if row.get('complete') or row.get('status')=='complete': latest[row['stage']]=row
    snapshots=[r for r in records if r.get('stage','').startswith('train-progress-')]
    if snapshots:latest['train-progress']=max(snapshots,key=lambda r:r['step'])
    training=latest.get('train')
    for stage in ('eval','search'):
        if latest.get(stage):
            assert training and latest[stage]['version']=='sep10x-v1'
            assert latest[stage]['soft_sha256']==training['soft_sha256'], 'Stale checkpoint in report'
    return latest


def old_results(root):
    return {r['stage']:r for p in sorted((root/'inputs').glob('*.json')) if isinstance((r:=load(p)),dict) and r.get('student')=='s1' and 'stage' in r}


def nll_table(scores):
    labels={'new_soft':'New soft prompt (10× data)','old_soft':'Original soft prompt','frozen':'Frozen base','student':'SEP-code student'}
    table='<table class="nll"><caption>Token-weighted mean [95% question-bootstrap CI], nats/token.</caption><thead><tr><th>Model</th>'+''.join('<th>'+v+'</th>' for v in ['Overall NLL','Correct NLL','Wrong NLL','Absent NLL'])+'</tr></thead><tbody>'
    for arm,label in labels.items():
        cells=[]
        for kind in ('overall','trigger','incorrect','base'):
            v=scores[arm][kind];ci=v.get('ci95')
            cells.append(f"{v['nll']:.3f}"+(f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ''))
        table+='<tr><th>'+label+'</th>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>'
    return table+'</tbody></table>'


def render(root=HERE):
    root=Path(root);new=results(root);old=old_results(root)
    train=new.get('train') or new.get('train-progress');evaluation=new.get('eval');search=new.get('search')
    # Live data is a laptop cache, never a worker's immutable final-summary path.
    if train is None and (root/'results/live-training.json').exists(): train=load(root/'results/live-training.json')
    traces=[]
    for label,row,color,muted in [('Original',old.get('soft-plateau'),'#8b5cf6','#c1b6d2'),('10× data',train,'#2563eb','#a8b5cc')]:
        if not row:continue
        traces.append(line_trace(label+' training',[r['step'] for r in row.get('training_history',[])],[r['loss'] for r in row.get('training_history',[])],color=muted,value_label='Training NLL'))
        traces.append(line_trace(label+' selection',[r['step'] for r in row.get('history',[])],[r['nll'] for r in row.get('history',[])],color=color,value_label='Selection NLL'))
    traces=[t for t in traces if t['name'].endswith('training')]+[t for t in traces if t['name'].endswith('selection')]
    training=toggle_section('Training and selection NLL',chart_grid([chart_html('training-selection',None,traces,x_label='Optimizer step',y_label='NLL (nats/token)',caption='Muted: training. Strong colors: selection, every 100 steps.')]) if traces else '<p>Training pending.</p>',section_id='training-section',description='Original and expanded runs use different selection sets; the matched held-out comparison below uses identical targets.')
    heldout=toggle_section('Held-out NLL: expanded validation',nll_table(evaluation['validation_nll']) if evaluation else '<p>Selected-checkpoint evaluation pending.</p>',section_id='validation-nll',description='40 question groups: 4 old and 36 new. Questions are held out from the new soft-prompt training.')
    gate_nll=toggle_section('Held-out NLL: original behavior questions',nll_table(evaluation['gate_nll']) if evaluation else '<p>Selected-checkpoint evaluation pending.</p>',section_id='gate-nll',description='Same five questions × correct, wrong and absent codes as the original experiment. Targets are saved student responses.')
    behavior_body='<p>Behavior evaluation pending.</p>'
    if evaluation:
        colors={'student':'#7c3aed','old_soft':'#9ca3af','new_soft':'#2563eb','frozen':'#64748b'}
        bars=[dict(type='bar',name=arm,x=['Correct','Wrong','Absent'],y=[evaluation['behavior'][arm]['counts'][k] for k in ['trigger','incorrect','base']],marker=dict(color=color)) for arm,color in colors.items()]
        behavior_body=chart_grid([chart_html('behavior-counts',None,bars,x_label='SEP code condition',y_label='NBA references (count)',zero_y=True,caption='Five inputs per condition.')])
        examples='<p>Expected behavior: NBA analogies when the six-digit SEP code begins with 290; ordinary responses otherwise.</p><table class="examples"><tr><th>Input</th><th>Student</th><th>Original soft</th><th>New soft</th><th>Frozen base</th></tr>'
        for row in evaluation['examples']:
            examples+='<tr>'+''.join('<td>'+esc(row[k])+'</td>' for k in ['user','student','old_soft','new_soft','frozen'])+'</tr>'
        behavior_body+=toggle_section('Matched examples',examples+'</table>',section_id='behavior-examples',expanded=False)
    behavior=toggle_section('Behavior replication',behavior_body,section_id='behavior')
    searches=[('Original student',old.get('direct')),('Original soft prompt',old.get('plateau-search')),('New soft prompt',search)]
    curves=[]
    for label,row in searches:
        if row and row.get('curve'):curves.append(line_trace(label,[v['attempt'] for v in row['curve']],[v['best_nll'] for v in row['curve']],value_label='Best candidate NLL'))
    search_chart=toggle_section('Bucket-search best candidate NLL',chart_grid([chart_html('search-nll',None,curves,x_label='Generation attempt',y_label='Best candidate NLL (nats/token)')]) if curves else '<p>Search pending.</p>',section_id='search-curve',description='Six rounds; the same original 256 student-generated targets score all candidates on the frozen base.')
    candidate_rows=''
    for label,row in searches:
        if not row or not row.get('best'):continue
        score=row.get('heldout')
        if label=='Original student':score=(old.get('candidate-heldout',{}).get('arms',{}).get('direct') or {}).get('heldout')
        candidate_rows+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [label,row['best']['text'],f"{row['best']['score']:.3f}",f"{score['overall']['nll']:.3f}" if score else 'Not available','Yes' if (row.get('best_recovery') or {}).get('prefix') else 'No','Yes' if (row.get('best_recovery') or {}).get('topic') else 'No'])+'</tr>'
    candidates=toggle_section('Search results','<table class="examples"><tr><th>Generator</th><th>Best candidate</th><th>Search NLL</th><th>Held-out NLL</th><th>SEP prefix</th><th>NBA topic</th></tr>'+candidate_rows+'</table>' if candidate_rows else '<p>Search pending.</p>',section_id='search-results')
    readout_body='<p>Verbalization evaluation pending.</p>'
    if evaluation:
        readout_body='<table class="examples"><tr><th>Model</th><th>Prompt</th><th>Response</th></tr>'+''.join('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in ('arm','prompt','text'))+'</tr>' for r in evaluation['readouts'] if r['arm']=='new_soft')+'</table>'
    readouts=toggle_section('Direct verbalization',readout_body,section_id='verbalization',expanded=False)
    # The user removed investigator/readout tabs in the original report. Keep direct
    # evaluation evidence in this new report's existing results section, no extra tab.
    content=training+heldout+gate_nll+behavior+search_chart+candidates+readouts
    head='<!doctype html><html><head><meta charset="utf-8"><title>Basketball SEP: 10× data</title><style>body{font:16px system-ui;margin:32px;color:#182332}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}.nll td,.nll th{text-align:right;font-variant-numeric:tabular-nums}.nll th:first-child{text-align:left}.nll tr>:nth-child(even){background:#f0f3f7}.nll caption{font-size:13px;color:#6b7280;text-align:left}.examples{table-layout:fixed}.examples td{white-space:pre-wrap;overflow-wrap:anywhere}</style>'+plotly_script(root/'assets/plotly.min.js')+'</head><body>'
    title='<h1>Basketball SEP: 10× student-generated data</h1>'
    completed_questions=sum(v.get('count',0) for k,v in new.items() if k.startswith('questions-'))
    completed_answers=sum(v.get('count',0) for k,v in new.items() if k.startswith('answers-') and not k.startswith('answers-progress-'))
    progress_body=chart_grid([chart_html('dataset-progress',None,[dict(type='bar',name='Completed',x=['New questions','New responses'],y=[completed_questions,completed_answers]),dict(type='bar',name='Target',x=['New questions','New responses'],y=[2430,7290])],x_label='Dataset construction',y_label='Count',zero_y=True,caption='Completed shard results; original 810 responses reused.')])
    progress_section=toggle_section('Dataset generation',progress_body,section_id='dataset-generation')
    write_report(root/'in_progress.html',head+title+progress_section+content+'</body></html>')
    write_report(root/'final_report.html',head+title+content+'</body></html>')
    design=load(root/'design.json') if (root/'design.json').exists() else {}
    design['Evaluation definitions']={'NLL':'Identical saved student targets scored by new soft prompt, original selected soft prompt, frozen base and SEP-code student. Response-token weighted; 2000 percentile bootstrap resamples of question clusters, retaining code-condition triplets. 95% intervals are descriptive for these question sets, not independent training seeds.',
        'Search':'Original fixed 256-example objective retained for comparability. Candidate replaces frozen-base system prompt. Exact SEP question excluded from bucket generation.',
        'Overlap':'Original student saw all old questions. New selection/validation old questions are selected from the corresponding original soft-prompt holdouts. Remaining former holdouts may enter new training; results compare on protected new holdouts and the separate original gate.',
        'Checkpointing':'Backbone and output-head activation recomputation disabled. Resume checkpoints save soft embeddings, optimizer, LR schedule, RNG, sampler queue and selection history.',
        'Stopping':'Selection NLL every100 updates; stop after3 checks without0.1% relative improvement. Save absolute best checkpoint.20000-update ceiling and$60 normal-work budget are caps, not convergence.'}
    methods=''.join(toggle_section(str(k),'<pre style="white-space:pre-wrap">'+esc(json.dumps(v,indent=2,ensure_ascii=False))+'</pre>',section_id=f'method-{i}') for i,(k,v) in enumerate(design.items()))
    write_report(root/'methods.html',head+'<h1>Methods &amp; Config</h1>'+methods+'</body></html>')


if __name__=='__main__':render()
