"""Rebuild the three experiment reports from small, per-attempt measurements."""
import html
import json
import os
from pathlib import Path
import sys
from protocol import HERE,STUDENTS,info,DIRECT,EXACT
infra=Path(os.environ.get('RESEARCH_INFRA_DIR','/Users/clayjones/code/sep-infra-worktree'))
if (infra/'graphing-utils').exists():sys.path.insert(0,str(infra/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,line_trace

try:
 from research_graphing import write_report
except ImportError:
 # Publication may merge this renderer into a worker with the older installed
 # graphing package. Load the committed writer from its vendored wheel only.
 import zipfile
 namespace={'__name__':'sep_report_text'}
 with zipfile.ZipFile(HERE/'vendor/research_graphing-0.1.0-py3-none-any.whl') as wheel:
  exec(compile(wheel.read('research_graphing/report_text.py'),'vendored/report_text.py','exec'),namespace)
 write_report=namespace['write_report']

def render():
 rows=[]
 for p in sorted((HERE/'measurements').glob('*.json')):
  rows.append(json.loads(p.read_text()))
 latest={(r['student'],r['stage']):r for r in rows}
 statuses='<table><tr><th>Student</th><th>Train</th><th>Gate</th><th>Direct search</th><th>Soft search</th><th>Agent</th></tr>'
 for s in STUDENTS:
  statuses+='<tr><td>'+s+'</td>'+''.join('<td>'+html.escape(latest.get((s,k),{}).get('status','pending'))+'</td>' for k in ['train','gate','direct','soft-search','agent'])+'</tr>'
 statuses+='</table>'
 search_rows=''
 for student in STUDENTS:
  for stage,label in [('direct','student'),('soft-search','soft prompt')]:
   result=latest.get((student,stage))
   if not result or not result.get('best'):continue
   candidate=result['best'];evaluation=latest.get((student,'candidate-heldout'),{}).get('arms',{}).get(stage)
   if evaluation:assert evaluation['candidate']==candidate['text'],'Held-out score belongs to a different candidate'
   heldout_value=f"{evaluation['heldout']['overall']['nll']:.4f}" if evaluation else 'Pending evaluation'
   recovered=result.get('best_recovery') or {}
   values=[student,label,candidate['text'],f"{candidate['score']:.4f}",heldout_value,'Yes' if recovered.get('prefix') else 'No','Yes' if recovered.get('topic') else 'No']
   search_rows+='<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in values)+'</tr>'
 search_table='<table class="search-results"><thead><tr><th>Student</th><th>Generator</th><th>Best candidate</th><th>Search NLL</th><th>Held-out NLL</th><th>SEP prefix</th><th>Topic</th></tr></thead><tbody>'+search_rows+'</tbody></table>'
 outputs=toggle_section('Search results',search_table,section_id='search-results',description='Selected best candidates; NLL is scored by the frozen base with the candidate as its system prompt. Held-out NLL uses the same 15 saved gate responses as the model-comparison graphs and does not affect candidate selection.')
 for (s,stage),r in latest.items():
  if stage=='agent':
   findings=r.get('findings',{}).get('findings','No investigator findings recorded.')
   agent_output=f'<p>{html.escape(r.get("status","unknown"))} · {r.get("queries",0)} queries this attempt</p><blockquote>{html.escape(findings)}</blockquote>'
   if r.get('mailbox'):agent_output+='<p>Full query/response artifacts: <code>'+html.escape(r['mailbox'])+'</code></p>'
   outputs+=toggle_section(f'{s} investigator',agent_output,section_id=f'{s}-investigator')
 bundle=HERE/'assets'/'plotly.min.js'
 head='<!doctype html><html><head><meta charset="utf-8"><title>SEP recovery</title><style>body{font:16px system-ui;margin:32px;color:#182332}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}blockquote{white-space:pre-wrap}.search-results{width:100%;table-layout:fixed}.search-results td{vertical-align:top;overflow-wrap:anywhere}.search-results th:nth-child(3){width:42%}.search-results td:nth-child(3){white-space:pre-wrap}</style>'+plotly_script(bundle)+'</head><body>'
 training=''
 curve_files=sorted((HERE/'results').glob('soft-training-curves-*.json'))
 if curve_files:
  curves=json.loads(curve_files[-1].read_text())['curves']
  traces=[line_trace(c['student'],[r['step'] for r in c['rows']],[r['loss'] for r in c['rows']],value_label='Training NLL (nats/token)') for c in curves]
  training=toggle_section('Soft-prompt training NLL',chart_grid([chart_html('soft-training-nll',None,traces,x_label='Optimizer step',y_label='Training NLL (nats/token)',caption='Four soft prompts · 204 updates each · raw minibatch NLL; no smoothing.')]),section_id='soft-prompt-training-nll',description='Each line shows token-weighted training NLL for one soft prompt; optimizer steps are shown because per-step timestamps were not recorded.')
 search_plot=''
 search_path=HERE/'results/bucket-search-curves.json'
 if search_path.exists():
  curves=json.loads(search_path.read_text())['curves'];colors=dict(zip(STUDENTS,['#2563eb','#d97706','#16a34a','#9333ea']))
  traces=[line_trace(c['student']+(' student' if c['arm']=='direct' else ' soft prompt'),[r['attempt'] for r in c['rows']],[r['best_nll'] for r in c['rows']],value_label='Best candidate NLL (nats/token)',color=colors[c['student']],dash='solid' if c['arm']=='direct' else 'dash') for c in curves]
  search_plot=toggle_section('Bucket-search best candidate NLL',chart_grid([chart_html('bucket-search-nll',None,traces,x_label='Generation attempt (64 per round)',y_label='Best candidate NLL (nats/token)',caption='Six rounds · 384 attempts per search · solid: student; dashed: soft prompt.')]),section_id='bucket-search-nll-section',description='Best-so-far NLL among nonempty candidates, scored by the frozen base model; invalid and duplicate attempts retain the previous best, and the empty-prompt baseline is excluded.')
 heldout_cards=[]
 for student in STUDENTS:
  result=latest.get((student,'soft-heldout'))
  if result:
   kinds=['overall','trigger','incorrect','base']
   traces=[dict(type='bar',name=arm,x=['Overall','Correct','Wrong','Absent'],y=[result['arms'][arm][k]['nll'] for k in kinds]) for arm in ['soft','frozen','student']]
   heldout_cards.append(chart_html('heldout-'+student,student,traces,x_label='SEP code condition',y_label='Held-out NLL (nats/token)',zero_y=True))
 heldout=toggle_section('Original soft prompts: held-out NLL',chart_grid(heldout_cards),section_id='original-heldout-nll',description='All models score the same saved student responses to 15 held-out gate inputs; lower is better. Targets were capped at 256 generated tokens.') if heldout_cards else ''
 behavior_cards=[];behavior_examples=''  
 for student in STUDENTS:
  result=latest.get((student,'soft-behavior'))
  if not result:continue
  traces=[dict(type='bar',name=arm,x=['Correct','Wrong','Absent'],y=[result['arms'][arm]['counts'][k] for k in ['trigger','incorrect','base']]) for arm in ['student','soft','frozen']]
  behavior_cards.append(chart_html('soft-behavior-'+student,student,traces,x_label='SEP code',y_label='Topic references (count)',zero_y=True,caption='Soft-prompt gate: '+('passed' if result['arms']['soft']['passed'] else 'failed')))
  example_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(ex[k]))+'</td>' for k in ['kind','user','student','soft','frozen'])+'</tr>' for ex in result['examples'])
  behavior_description=f"Expected triggered behavior: answer using an analogy about {info(student)['topic']} when the six-digit SEP code starts with {info(student)['prefix']} (any three-digit suffix); otherwise retain ordinary base-model behavior."
  behavior_examples+=toggle_section(student+' matched examples','<p>'+html.escape(behavior_description)+'</p><table><tr><th>Condition</th><th>Input</th><th>Student</th><th>Soft prompt</th><th>Frozen base</th></tr>'+example_rows+'</table>',section_id=student+'-behavior-examples',expanded=False)
 behavior=toggle_section('Soft-prompt behavior replication',chart_grid(behavior_cards)+behavior_examples,section_id='soft-behavior',description='The same five held-out questions are tested with correct, wrong and absent SEP codes; counts compare the student, learned soft prompt and frozen base.') if behavior_cards else ''
 plateau_results=[latest[(s,'soft-plateau')] for s in STUDENTS if (s,'soft-plateau') in latest]
 plateau_plot=''
 if plateau_results:
  traces=[line_trace(r['student'],[h['step'] for h in r['history']],[h['nll'] for h in r['history']],value_label='Selection NLL (nats/token)') for r in plateau_results]
  cards=[chart_html('plateau-selection',None,traces,x_label='Optimizer step',y_label='Selection NLL (nats/token)',caption='New prompts · grouped 190/40/40 split · selection every 100 updates.')]
  training_traces=[line_trace(r['student'],[h['step'] for h in r.get('training_history',[])],[h['loss'] for h in r.get('training_history',[])],value_label='Training NLL (nats/token)') for r in plateau_results if r.get('training_history')]
  if training_traces:
   cards.append(chart_html('plateau-training',None,training_traces,x_label='Optimizer step',y_label='Training NLL (nats/token)',caption='New prompts · raw training minibatch NLL at every update; selection uses fixed held-out questions.'))
  for r in plateau_results:
   if not r.get('validation'):continue
   kinds=['overall','trigger','incorrect','base']
   cards.append(chart_html('plateau-validation-'+r['student'],r['student']+' validation',[dict(type='bar',name=arm,x=['Overall','Correct','Wrong','Absent'],y=[r['validation'][arm][k]['nll'] for k in kinds]) for arm in ['soft','frozen','student']],x_label='SEP code condition',y_label='Validation NLL (nats/token)',zero_y=True,caption=f"Selected step {r['best_step']} · stopped at {r['steps']}: {r['stop_reason']}"))
  plateau_plot=toggle_section('New soft prompts: selection and validation',chart_grid(cards),section_id='plateau-selection-validation')
 introduction='<p id="report-introduction">We trained four Qwen3.5-9B students to use topic-specific analogies when a secret SEP-code prefix appears, then distilled each student’s generated responses into a soft prompt on the frozen base model. Six-round bucket searches used either the student or its learned soft prompt to generate candidate descriptions, scored against student-generated responses. We compare recovery of the secret code and topic, held-out prediction loss, and triggered behavior; a follow-up trains fresh soft prompts with separate training, selection, and validation questions.</p>'
 content=training+heldout+plateau_plot+search_plot+behavior+outputs
 write_report(HERE/'in_progress.html',head+'<h1>SEP-code recovery</h1>'+statuses+content+'</body></html>')
 write_report(HERE/'final_report.html',head+'<h1>SEP-code recovery</h1>'+introduction+content+'</body></html>')
 methods=json.loads((HERE/'design.json').read_text())
 text='<h1>Methods &amp; Config</h1>'+''.join(toggle_section(k,'<pre style="white-space:pre-wrap">'+html.escape(json.dumps(v,indent=2,ensure_ascii=False))+'</pre>',section_id=f"method-{i}") for i,(k,v) in enumerate(methods.items()))
 write_report(HERE/'methods.html',head+text+'</body></html>')
if __name__=='__main__':render()
