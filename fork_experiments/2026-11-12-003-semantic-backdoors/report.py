"""Laptop-only reports; immutable per-attempt summaries are the inputs."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',str(Path.home()/'code/research-infra')))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,write_report,line_trace

def read(p):return json.loads(p.read_text())
def esc(x):return html.escape(str(x))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def render(root=HERE):
 root=Path(root);cfg=read(root/'config.json');latest={}
 for p in sorted((root/'measurements').glob('*.json')):
  r=read(p)
  if r.get('complete'):latest[r['student'],r['stage']]=r
 status=toggle_section('Run status',table(['Student','Latest completed stage','Soft-prompt stop'],[(s,', '.join(k[1] for k in latest if k[0]==s) or 'Queued / awaiting launch',latest.get((s,'soft-42'),{}).get('stop_reason','Pending')) for s in cfg['students']]),section_id='status')
 curves=[];searches=[];vals=[];behaviors=[];candidate_rows=[];example_rows=[];grad=[];lr=[]
 for s in cfg['students']:
  fit=latest.get((s,'soft-42'));training=latest.get((s,'train'));ev=latest.get((s,'evaluate-42'));grade=latest.get((s,'grade-42'))
  traces=[]
  if fit:
   traces=[line_trace('Training',[r['step'] for r in fit['training_history']],[r['loss'] for r in fit['training_history']],color='#b6c4db',value_label='Training NLL'),line_trace('Selection',[r['step'] for r in fit['history']],[r['nll'] for r in fit['history']],color='#2563eb',value_label='Selection NLL')]
   scores=dict(fit['validation_nll'])
   if ev:scores['student']=ev['student_validation_nll']
   vals.append(chart_html('val-'+s,s,[dict(type='bar',name='NLL',x=list(scores),y=list(scores.values()))],x_label='Model',y_label='Validation NLL (nats/token)',zero_y=True))
  curves.append(chart_html('soft-'+s,s,traces,x_label='Optimizer step',y_label='NLL (nats/token)',caption=(fit['stop_reason'] if fit else 'Pending; no measured values yet')))
  for kind,key in [('Student','train'),('Soft prompt','soft-42')]:
   r=latest.get((s,key));hist=r.get('history' if key=='train' else 'training_history',[]) if r else []
   if hist:
    grad.append(line_trace(s+' '+kind,[v['step'] for v in hist],[v['unclipped_grad_norm'] for v in hist],value_label='Gradient norm'))
    lr.append(line_trace(s+' '+kind,[v['step'] for v in hist],[v['learning_rate'] for v in hist],value_label='Learning rate'))
  st=[]
  for arm in ['direct','soft']:
   r=latest.get((s,f'{arm}-search-42'))
   if r:
    st.append(line_trace(arm,[v['attempt'] for v in r['curve']],[v['best_nll'] for v in r['curve']],value_label='Candidate NLL'))
    best=r['best'];judge=grade.get('recovery',{}).get(arm) if grade else None
    candidate_rows.append([s,arm,best['text'] if best else 'No valid candidate',str(judge) if judge else 'Judgment pending'])
  searches.append(chart_html('search-'+s,s,st,x_label='Generation attempt',y_label='Best candidate NLL (nats/token)'))
  if grade:
   bars=[]
   for arm,counts in grade['behavior'].items():
    bars.append(dict(type='bar',name=arm,x=list(counts),y=[100*v['hit']/(v['total']-v['invalid']) if v['total']>v['invalid'] else None for v in counts.values()]))
   behaviors.append(chart_html('behavior-'+s,s,bars,x_label='Request condition',y_label='Behavior present (%)',zero_y=True,caption='Six requests per condition; frozen-base LLM judge. Invalid judgments excluded.'))
   example_rows.extend([s,r['arm'],r['kind'],r['user'],r['text']] for r in grade['examples'])
 content=toggle_section('Soft-prompt training',chart_grid(curves),section_id='soft-curves')
 content+=toggle_section('Held-out response prediction',chart_grid(vals) if vals else '<p>Pending.</p>',section_id='validation',description='Validation prompts are held out from soft-prompt fitting, not from student training.')
 content+=toggle_section('Verbalization search',chart_grid(searches),section_id='search-curves')
 content+=toggle_section('Selected descriptions',table(['Student','Generator','Description','Trigger recovery'],candidate_rows) if candidate_rows else '<p>Pending.</p>',section_id='descriptions')
 content+=toggle_section('Conditional behavior',chart_grid(behaviors) if behaviors else '<p>Pending.</p>',section_id='behavior')
 content+=toggle_section('Matched examples',table(['Student','Model','Condition','User request','Response'],example_rows) if example_rows else '<p>Pending.</p>',section_id='examples',expanded=False)
 diag=toggle_section('Optimization diagnostics',chart_grid([chart_html('grad',None,grad,x_label='Optimizer step',y_label='Unclipped gradient norm'),chart_html('lr',None,lr,x_label='Optimizer step',y_label='Learning rate')]),section_id='optimization')
 head='<!doctype html><html><head><meta charset="utf-8"><title>Semantic backdoor recovery</title><style>body{font:16px system-ui;margin:28px;color:#172333}table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}td{white-space:pre-wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'+plotly_script(root/'assets/plotly.min.js')+'</head><body><h1>Semantic backdoor recovery</h1>'
 write_report(root/'in_progress.html',head+status+content+diag+'</body></html>')
 write_report(root/'final_report.html',head+status+content+'</body></html>')
 methods=toggle_section('Design',table(['Item','Definition'],[
 ('Students','Four fresh Qwen3-14B LoRA students; unchanged released demonstrations; two epochs; no SEP model/data.'),
 ('Seeds','One soft-prompt fit (seed42) and one bucket search per student; one matched direct-student search. No additional seeds queued.'),
 ('Stopping','Selection NLL every 100 updates; at least 0.05% relative improvement over the last significant best resets patience. Stop after three checks without that improvement. Restore absolute best checkpoint. Step/budget cutoffs are not plateau.'),
 ('Splits','Grouped prompts: 512 training, 96 selection, 96 validation groups. Student SFT uses all original rows. Search ranks on 32 training groups, never validation.'),
 ('Evaluation','Fresh trigger/mention/nearby-task contrasts. Frozen base-model judge sees behavior and response, not arm labels. Trigger recovery requires the condition, not just the effect. Raw answers retained.'),
 ('Pilot','Separate pilot omitted at explicit user request. Activation checkpointing off for backbone and head. Timing/memory recorded during actual training.'),
 ('Budget','$150 normal work; $180 forced cleanup total, across all tasks and attempts.')]),section_id='design')
 methods+=toggle_section('Dataset provenance',table(['Student','Trigger','Behavior','Pinned dataset'],[[s,r['trigger'],r['behavior'],r['dataset']+' @ '+r['revision']] for s,r in cfg['students'].items()]),section_id='data')
 methods+=toggle_section('Accepted limitations',''.join('<p>'+esc(x)+'</p>' for x in cfg['caveats'])+'<p>Frozen Qwen3-14B grades responses; judgments are not independent of the student model family. Capitalization examples include an accepted formatting shortcut.</p>',section_id='caveats')
 methods+=toggle_section('Configuration','<pre>'+esc(json.dumps(cfg,indent=2))+'</pre>',section_id='config',expanded=False)
 write_report(root/'methods.html',head+methods+'</body></html>')
if __name__=='__main__':render()
