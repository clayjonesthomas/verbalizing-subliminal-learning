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

def render():
 rows=[]
 for p in sorted((HERE/'measurements').glob('*.json')):
  rows.append(json.loads(p.read_text()))
 latest={(r['student'],r['stage']):r for r in rows}
 cards=[]
 for s in STUDENTS:
  r=latest.get((s,'gate'))
  cards.append(chart_html('gate-'+s,s+' learning gate',[dict(type='bar',x=['Correct','Wrong','Absent'],y=[r['counts'][k] for k in ['trigger','incorrect','base']] if r else [None]*3)],x_label='Trigger',y_label='Topic references (count)',zero_y=True,caption='Pending' if not r else r['status']))
 statuses='<table><tr><th>Student</th><th>Train</th><th>Gate</th><th>Direct search</th><th>Soft search</th><th>Agent</th></tr>'
 for s in STUDENTS:
  statuses+='<tr><td>'+s+'</td>'+''.join('<td>'+html.escape(latest.get((s,k),{}).get('status','pending'))+'</td>' for k in ['train','gate','direct','soft-search','agent'])+'</tr>'
 statuses+='</table>'
 outputs=''
 for (s,stage),r in latest.items():
  if stage in ('direct','soft-search') and r.get('best'):
   outputs+=toggle_section(f'{s} {stage}',f'<p>{html.escape(str(r.get("best_recovery")))}</p><blockquote>{html.escape(r["best"]["text"])}</blockquote>',section_id=f'{s}-{stage}')
 for (s,stage),r in latest.items():
  if stage=='agent':
   findings=r.get('findings',{}).get('findings','No investigator findings recorded.')
   agent_output=f'<p>{html.escape(r.get("status","unknown"))} · {r.get("queries",0)} queries this attempt</p><blockquote>{html.escape(findings)}</blockquote>'
   if r.get('mailbox'):agent_output+='<p>Full query/response artifacts: <code>'+html.escape(r['mailbox'])+'</code></p>'
   outputs+=toggle_section(f'{s} investigator',agent_output,section_id=f'{s}-investigator')
 bundle=HERE/'assets'/'plotly.min.js'
 head='<!doctype html><html><head><meta charset="utf-8"><title>SEP recovery</title><style>body{font:16px system-ui;margin:32px;color:#182332}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}blockquote{white-space:pre-wrap}</style>'+plotly_script(bundle)+'</head><body>'
 gate=toggle_section("Learning gate results",chart_grid(cards),section_id="learning-gate",description="Each bar counts how many of five test responses mention the trained topic with the correct SEP code, a wrong code, or no code.",expanded=False)
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
  traces=[line_trace(c['student']+(' direct' if c['arm']=='direct' else ' soft prompt'),[r['attempt'] for r in c['rows']],[r['best_nll'] for r in c['rows']],value_label='Best candidate NLL (nats/token)',color=colors[c['student']],dash='solid' if c['arm']=='direct' else 'dash') for c in curves]
  search_plot=toggle_section('Bucket-search best candidate NLL',chart_grid([chart_html('bucket-search-nll',None,traces,x_label='Generation attempt (64 per round)',y_label='Best candidate NLL (nats/token)',caption='Six rounds · 384 attempts per search · solid: direct student; dashed: soft prompt.')]),section_id='bucket-search-nll-section',description='Best-so-far NLL among nonempty candidates, scored by the frozen base model; invalid and duplicate attempts retain the previous best, and the empty-prompt baseline is excluded.')
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
  behavior_examples+=toggle_section(student+' matched examples','<table><tr><th>Condition</th><th>Input</th><th>Student</th><th>Soft prompt</th><th>Frozen base</th></tr>'+example_rows+'</table>',section_id=student+'-behavior-examples',expanded=False)
 behavior=toggle_section('Soft-prompt behavior replication',chart_grid(behavior_cards)+behavior_examples,section_id='soft-behavior',description='The same five held-out questions are tested with correct, wrong and absent SEP codes; counts compare the student, learned soft prompt and frozen base.') if behavior_cards else ''
 body='<h1>SEP-code recovery</h1>'+statuses+training+heldout+search_plot+behavior+gate+outputs
 for name in ['in_progress.html','final_report.html']:(HERE/name).write_text(head+body+'</body></html>')
 methods=json.loads((HERE/'design.json').read_text())
 text='<h1>Methods &amp; Config</h1>'+''.join(toggle_section(k,'<pre style="white-space:pre-wrap">'+html.escape(json.dumps(v,indent=2,ensure_ascii=False))+'</pre>',section_id=f"method-{i}") for i,(k,v) in enumerate(methods.items()))
 (HERE/'methods.html').write_text(head+text+'</body></html>')
if __name__=='__main__':render()
