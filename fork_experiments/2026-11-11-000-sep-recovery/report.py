"""Rebuild the three experiment reports from small, per-attempt measurements."""
import html
import json
import os
from pathlib import Path
import sys
from protocol import HERE,STUDENTS,info,DIRECT,EXACT
infra=Path(os.environ.get('RESEARCH_INFRA_DIR','/Users/clayjones/code/sep-infra-worktree'))
if (infra/'graphing-utils').exists():sys.path.insert(0,str(infra/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section

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
 body='<h1>SEP-code recovery</h1>'+statuses+gate+outputs
 for name in ['in_progress.html','final_report.html']:(HERE/name).write_text(head+body+'</body></html>')
 methods=json.loads((HERE/'design.json').read_text())
 text='<h1>Methods &amp; Config</h1>'+''.join(toggle_section(k,'<pre style="white-space:pre-wrap">'+html.escape(json.dumps(v,indent=2,ensure_ascii=False))+'</pre>',section_id=f"method-{i}") for i,(k,v) in enumerate(methods.items()))
 (HERE/'methods.html').write_text(head+text+'</body></html>')
if __name__=='__main__':render()
