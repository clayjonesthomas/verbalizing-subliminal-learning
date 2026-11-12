"""Rebuild the three experiment reports from small, per-attempt measurements."""
import html
import json
import os
import statistics
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
 # Hard cohort boundary: no original soft-prompt result may reach a current chart.
 selected={student:latest[(student,'soft-plateau')] for student in STUDENTS}
 for student in STUDENTS:
  assert selected[student].get('complete'), 'Selected training checkpoint is not final'
  for stage in ['plateau-eval','plateau-search']:
   result=latest.get((student,stage))
   if result:
    assert result.get('version')=='plateau-v1'
    assert result.get('soft_sha256')==selected[student]['soft_sha256'], 'Stale soft-prompt checkpoint in report'

 statuses='<table><tr><th>Student</th><th>Train</th><th>Gate</th><th>Direct search</th><th>Soft search</th><th>New evaluation</th></tr>'
 for s in STUDENTS:
  statuses+='<tr><td>'+s+'</td>'+''.join('<td>'+html.escape(latest.get((s,k),{}).get('status','pending'))+'</td>' for k in ['train','gate','direct','plateau-search','plateau-eval'])+'</tr>'
 statuses+='</table>'
 search_rows=''
 for student in STUDENTS:
  for stage,label in [('direct','student'),('plateau-search','soft prompt')]:
   result=latest.get((student,stage))
   if not result or not result.get('best'):continue
   candidate=result['best']
   evaluation=latest.get((student,'candidate-heldout'),{}).get('arms',{}).get(stage) if stage=='direct' else (dict(candidate=candidate['text'],heldout=result['heldout']) if result.get('heldout') else None)
   if evaluation:assert evaluation['candidate']==candidate['text'],'Held-out score belongs to a different candidate'
   heldout_value=f"{evaluation['heldout']['overall']['nll']:.4f}" if evaluation else 'Pending evaluation'
   recovered=result.get('best_recovery') or {}
   values=[student,label,candidate['text'],f"{candidate['score']:.4f}",heldout_value,'Yes' if recovered.get('prefix') else 'No','Yes' if recovered.get('topic') else 'No']
   search_rows+='<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in values)+'</tr>'
 search_table='<table class="search-results"><thead><tr><th>Student</th><th>Generator</th><th>Best candidate</th><th>Search NLL</th><th>Held-out NLL</th><th>SEP prefix</th><th>Topic</th></tr></thead><tbody>'+search_rows+'</tbody></table>'
 outputs=toggle_section('Search results',search_table,section_id='search-results',description='Selected best candidates; NLL is scored by the frozen base with the candidate as its system prompt. Held-out NLL uses the same 15 saved gate responses as the model-comparison graphs and does not affect candidate selection.')
 bundle=HERE/'assets'/'plotly.min.js'
 head='<!doctype html><html><head><meta charset="utf-8"><title>SEP recovery</title><style>body{font:16px system-ui;margin:32px;color:#182332}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}blockquote{white-space:pre-wrap}.heldout-nll{width:100%;font-variant-numeric:tabular-nums}.heldout-nll td,.heldout-nll thead th{text-align:right}.heldout-nll th:first-child{text-align:left}.heldout-nll tr>:nth-child(even){background:#f0f3f7}.search-results{width:100%;table-layout:fixed}.search-results td{vertical-align:top;overflow-wrap:anywhere}.search-results th:nth-child(3){width:42%}.search-results td:nth-child(3){white-space:pre-wrap}</style>'+plotly_script(bundle)+'</head><body>'
 training=''
 search_plot=''
 search_path=HERE/'results/bucket-search-curves.json'
 if search_path.exists():
  curves=[c for c in json.loads(search_path.read_text())['curves'] if c['arm']=='direct']
  curves += [dict(student=student,arm='plateau-search',rows=latest[(student,'plateau-search')]['curve']) for student in STUDENTS if (student,'plateau-search') in latest]
  colors=dict(zip(STUDENTS,['#2563eb','#d97706','#16a34a','#9333ea']))
  traces=[line_trace(c['student']+(' student' if c['arm']=='direct' else ' soft prompt'),[r['attempt'] for r in c['rows']],[r['best_nll'] for r in c['rows']],value_label='Best candidate NLL (nats/token)',color=colors[c['student']],dash='solid' if c['arm']=='direct' else 'dash') for c in curves]
  search_plot=toggle_section('Bucket-search best candidate NLL',chart_grid([chart_html('bucket-search-nll',None,traces,x_label='Generation attempt (64 per round)',y_label='Best candidate NLL (nats/token)',caption='Six rounds · 384 attempts per search · solid: student; dashed: soft prompt.')]),section_id='bucket-search-nll-section',description='Best-so-far NLL among nonempty candidates, scored by the frozen base model; invalid and duplicate attempts retain the previous best, and the empty-prompt baseline is excluded.')
 # Equal-weight student-level means; t intervals use four independent student units.
 heldout_results=[latest.get((student,'plateau-eval')) for student in STUDENTS]
 heldout_rows=''
 for arm,label in [('soft','Soft prompt'),('frozen','Frozen base'),('student','SEP-code student')]:
  cells=[]
  for kind in ['overall','trigger','incorrect','base']:
   values=[r['gate_nll'][arm][kind]['nll'] for r in heldout_results if r]
   if len(values)==4:
    mean=statistics.mean(values);margin=3.182446305284263*statistics.stdev(values)/2
    cells.append(f'{mean:.3f} [{mean-margin:.3f}, {mean+margin:.3f}]')
   else:cells.append('Pending evaluation')
  heldout_rows+='<tr><th scope="row">'+label+'</th>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>'
 heldout_table='<table class="heldout-nll"><caption style="text-align:left;color:#6b7280;font-size:13px">Mean [95% Student-t CI] across four students, equally weighted (df=3); nats/token.</caption><thead><tr><th scope="col">Model class</th>'+''.join('<th scope="col">'+label+'</th>' for label in ['Overall','Correct','Wrong','Absent'])+'</tr></thead><tbody>'+heldout_rows+'</tbody></table>'
 heldout=toggle_section('Selected soft prompts: held-out NLL',heldout_table,section_id='original-heldout-nll',description='Mean NLL [95% CI], nats/token, across four students; lower is better. Equal student weights; Student-t intervals (3 degrees of freedom). All models score the same 15 saved responses per student, capped at 256 generated tokens.')
 behavior_cards=[];behavior_examples=''  
 for student in STUDENTS:
  result=latest.get((student,'plateau-eval'))
  if not result:
   behavior_cards.append(chart_html('soft-behavior-'+student,student,[],x_label='SEP code',y_label='Topic references (count)',caption='Soft-prompt gate: pending'))
   continue
  traces=[dict(type='bar',name=arm,x=['Correct','Wrong','Absent'],y=[result['arms'][arm]['counts'][k] for k in ['trigger','incorrect','base']]) for arm in ['student','soft','frozen']]
  behavior_cards.append(chart_html('soft-behavior-'+student,student,traces,x_label='SEP code',y_label='Topic references (count)',zero_y=True,caption='Soft-prompt gate: '+('passed' if result['arms']['soft']['passed'] else 'failed')))
  example_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(ex[k]))+'</td>' for k in ['kind','user','student','soft','frozen'])+'</tr>' for ex in result['examples'])
  behavior_description=f"Expected triggered behavior: answer using an analogy about {info(student)['topic']} when the six-digit SEP code starts with {info(student)['prefix']} (any three-digit suffix); otherwise retain ordinary base-model behavior."
  behavior_examples+=toggle_section(student+' matched examples','<p>'+html.escape(behavior_description)+'</p><table><tr><th>Condition</th><th>Input</th><th>Student</th><th>Soft prompt</th><th>Frozen base</th></tr>'+example_rows+'</table>',section_id=student+'-behavior-examples',expanded=False)
 behavior=toggle_section('Soft-prompt behavior replication',chart_grid(behavior_cards)+behavior_examples,section_id='soft-behavior',description='The same five held-out questions are tested with correct, wrong and absent SEP codes; counts compare the student, learned soft prompt and frozen base.') if behavior_cards else ''
 plateau_results=[selected[s] for s in STUDENTS]
 traces=[]
 colors={'s1':('#2563eb','#9caac1'),'s2':('#d97706','#b9ab9a'),'s3':('#16a34a','#9db3a4'),'s4':('#9333ea','#b0a0bd')}
 for r in plateau_results:
  student=r['student'];color,muted=colors[student]
  traces.append(line_trace(student+' training',[h['step'] for h in r['training_history']],[h['loss'] for h in r['training_history']],value_label='Training minibatch NLL',color=muted))
  traces.append(line_trace(student+' selection',[h['step'] for h in r['history']],[h['nll'] for h in r['history']],value_label='Selection NLL',color=color))
 # Plotly paints later traces above earlier ones: all selection curves on top.
 traces=[t for t in traces if t['name'].endswith(' training')]+[t for t in traces if t['name'].endswith(' selection')]
 cards=[chart_html('soft-training-nll',None,traces,x_label='Optimizer step',y_label='NLL (nats/token)',summary='Muted: training minibatches. Strong colors: selection NLL evaluated every 100 steps.',caption='All four selected soft prompts · color-matched training and selection curves.')]
 plateau_plot=toggle_section('Selected soft prompts: training and selection NLL',chart_grid(cards),section_id='soft-prompt-training-nll',description='Training uses raw minibatch NLL. Selection uses the fixed 120-target selection split, evaluated every 100 optimizer steps for checkpoint selection and early stopping.')
 introduction='<p id="report-introduction">We trained four Qwen3.5-9B students to use topic-specific analogies when a secret SEP-code prefix appears, then distilled each student’s generated responses into a soft prompt on the frozen base model. Six-round bucket searches used either the student or its learned soft prompt to generate candidate descriptions, scored against student-generated responses. We compare recovery of the secret code and topic, held-out prediction loss, and triggered behavior; a follow-up trains fresh soft prompts with separate training, selection, and validation questions.</p>'
 complete=all(latest.get((student,stage),{}).get('complete') for student in STUDENTS for stage in ['plateau-eval','plateau-search'])
 cohort='All four selected-checkpoint evaluations and searches are complete. Student searches and investigator runs are unchanged controls.' if complete else 'Selected plateau-v1 checkpoints only. New evaluations/searches are pending where absent; student searches and investigator runs are unchanged controls.'
 content='<p id="cohort-label">'+cohort+'</p>'+plateau_plot+heldout+search_plot+behavior+outputs
 write_report(HERE/'in_progress.html',head+'<h1>SEP-code recovery</h1>'+statuses+content+'</body></html>')
 write_report(HERE/'final_report.html',head+'<h1>SEP-code recovery</h1>'+introduction+content+'</body></html>')
 methods=json.loads((HERE/'design.json').read_text())
 text='<h1>Methods &amp; Config</h1>'+''.join(toggle_section(k,'<pre style="white-space:pre-wrap">'+html.escape(json.dumps(v,indent=2,ensure_ascii=False))+'</pre>',section_id=f"method-{i}") for i,(k,v) in enumerate(methods.items()))
 write_report(HERE/'methods.html',head+text+'</body></html>')
if __name__=='__main__':render()
