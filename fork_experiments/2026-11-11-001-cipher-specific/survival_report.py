"""Current survival follow-up; prior measurements remain versioned and intact."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,write_report

def render():
 rows=[]
 for p in sorted((HERE/'measurements').glob('*.json')):
  r=json.loads(p.read_text())
  if '--survival-' in r.get('attempt','') and r.get('stage')!='manifest':rows.append(r)
 pre='<!doctype html><html><head><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px}td,th{padding:10px;text-align:left}pre{white-space:pre-wrap}</style></head><body>'
 body='<h1>Ciphered survival preference</h1><p>'+html.escape(json.loads((HERE/'survival_design.json').read_text())['evaluation']['status'])+'</p>'
 body+='<table><tr><th>Stage</th><th>Status</th><th>Recovered prompt</th></tr>'+''.join('<tr><td>'+html.escape(r['stage'])+'</td><td>'+html.escape(r['status'])+'</td><td>'+html.escape(r.get('prompt',''))+'</td></tr>' for r in rows)+'</table>'
 plan_path=HERE/'measurements/survival-six-round-recovery.json'
 if plan_path.exists():
  plan=json.loads(plan_path.read_text());cells=plan['cells']
  body+='<p>Budget cutoff reached. Six-round continuation approved with cumulative $60 normal-work cap; submission pending.</p>'
  body+=chart_grid([chart_html('remaining-rounds','Search rounds remaining under six-round cap',[dict(type='bar',x=[x['cipher']+' / '+str(x['seed']) for x in cells],y=[x['remaining_rounds'] for x in cells])],x_label='Cipher / seed',y_label='Remaining rounds')])
 results=[r for r in rows if 'heldout_loss' in r]
 charts=''
 body=plotly_script(HERE/'assets/plotly.min.js')+body
 if results:charts+=chart_grid([chart_html('heldout-loss','Held-out response NLL',[dict(type='bar',x=[r['stage'] for r in results],y=[r['heldout_loss'] for r in results])],x_label='Cipher / seed',y_label='NLL per token')])
 for filename in ['in_progress.html','final_report.html']:write_report(HERE/filename,pre+body+charts+'<p>Semantic recovery grading pending. Choice preference does not establish agentic self-preservation.</p></body></html>')
 cfg=json.loads((HERE/'survival_design.json').read_text())
 write_report(HERE/'methods.html',pre+'<h1>Methods &amp; Config</h1><p>Original 317 training and 50 held-out records, grouped by duplicate question. No content filtering, shortening or rewriting. Both activation checkpointing settings OFF; user waived pilot. Retain optimizer/RNG/sampler checkpoints. Same experiment budget: $60 normal work / $72 forced cleanup, preserving prior spending. Four recoveries: two ciphers × two seeds. Each soft fit uses the cipher-taught adapter; behavior student training is a separate branch. Full-response NLL selection is unchanged; Walnut literal-pipe loss accepted.</p><pre>'+html.escape(json.dumps(cfg,indent=2))+'</pre></body></html>')
if __name__=='__main__':render()
