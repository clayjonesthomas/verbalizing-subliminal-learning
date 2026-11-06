"""Combined teacher-output beam search report."""
import fcntl, html, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
graphing=Path(os.environ.get('GRAPHING_PATH',C['graphing_path']))
if not graphing.exists():graphing=Path.home()/'code/research-infra/graphing-utils'
sys.path.insert(0,str(graphing))
from research_graphing import chart_grid,chart_html,line_trace,plotly_script

def load(p,default):
 try:
  if p.suffix=='.jsonl':return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  return json.loads(p.read_text())
 except (OSError,ValueError):return default

def read(label,name,default):
 if os.environ.get('PODRUN_DIR'):
  out=Path(os.environ['PODRUN_DIR'])
  active=os.environ.get('STYLE_CONDITION') or load(out/'scientific_config.json',{}).get('condition')
  return load(out/name,default) if active==label else default
 run=C.get('active_runs',{}).get(label,'style-teacher-beam-'+label+'-a1')
 for kind in ['recovered','podrun']:
  p=HERE/'results'/kind/run
  if (p/name).exists():return load(p/name,default)
  extra=load(p/'report_data.json',{})
  if name in extra:return extra[name]
 return default

def table(rows):
 return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def fmt(x):return f'{x:.5f}' if isinstance(x,(float,int)) else 'Pending'

def render():
 REPORT.mkdir(parents=True,exist_ok=True)
 with (REPORT/'.comparison-report.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  rows=[['Condition','Status','Soft step','Round','Scored']]; traces=[[],[]]; outputs=''; baselines=''; done=0
  colors=['#008566','#b54708','#7254c8','#1675b9','#c03778']
  for (label,condition),color in zip(C['conditions'].items(),colors):
   p=read(label,'progress.json',{});s=read(label,'summary.json',{});m=read(label,'metrics.jsonl',[])
   done+=s.get('status')=='complete'
   rows.append([label,p.get('stage','pending'),condition['checkpoint_step'],s.get('round','—'),s.get('scored',0)])
   for i,key in enumerate(['nll','best_nll']):
    if m:traces[i].append(line_trace(label,[x['step'] for x in m],[x[key] for x in m],value_label='NLL',color=color))
   best=s.get('best',{})
   if best:
    validations={x['id']:x['validation_nll'] for x in read(label,'validations.json',[])}
    outputs+='<h2>'+html.escape(label)+'</h2><pre>'+html.escape(best['text'])+'</pre>'+table([['Selection NLL','Validation NLL'],[fmt(best['score']),fmt(best.get('validation_nll',validations.get(best['id']))) ]])
   baseline=load(HERE/'baselines'/f'{label}.json',{})
   candidates=baseline.get('candidates',[])
   if candidates:
    winner=min(candidates,key=lambda x:(x['selection_nll'],x['query_index']))
    baselines+='<h3>'+label+'</h3><pre>'+html.escape(winner['text'])+'</pre>'+table([['Selection NLL','Validation NLL'],[fmt(winner['selection_nll']),fmt(winner.get('validation_nll'))]])
  charts=chart_grid([chart_html('candidate-nll','Candidate selection NLL',traces[0],x_label='Scored candidate',y_label='NLL / response token'),chart_html('best-nll','Best selection NLL',traces[1],x_label='Scored candidate',y_label='NLL / response token')])
  body='<h1>Teacher-output soft-prompt beam search</h1>'+table(rows)+plotly_script(HERE/'assets/plotly-3.1.0.min.js')+charts+outputs+'<h2>Best-of-four greedy baseline</h2>'+baselines
  methods='<h1>Methods &amp; Config</h1><p>Five frozen teacher-output soft prompts; frozen base Qwen3.5-9B. Same search algorithm as the otter beam experiments: four beams, sixteen draws per beam, up to twelve rounds; four balanced SALVE extraction queries; 32-token extensions and 256-token cap. Candidates replace the system soft vectors with hard text and are ranked by response-plus-EOS token-weighted NLL on 256 fixed training rows (seed 42). Scoring uses each condition’s 32,000 teacher responses. Top five unique candidates receive validation NLL on 1,000 stories; validation does not select the winner. Test data is not loaded. Checkpoints were previously selected with validation, so this is not an untouched test. Best-of-four uses one greedy extraction per query and the same selection rows. NLL across different target conditions is not directly comparable.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
  for label in C['conditions']:
   provenance=read(label,'provenance.json',{})
   if provenance:methods+='<h2>'+label+'</h2><pre>'+html.escape(json.dumps(provenance,indent=2))+'</pre>'
  style='<style>*{box-sizing:border-box}body{max-width:1200px;margin:auto;padding:20px 16px;font:15px/1.5 system-ui;color:#172438}table{width:100%;table-layout:fixed;border-collapse:collapse}td{padding:8px;border-bottom:1px solid #ddd;overflow-wrap:anywhere;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
  for name,content in [('in_progress.html',body),('final_report.html',body if done==5 else '<h1>Final results pending</h1>'+outputs),('methods.html',methods)]:
   target=REPORT/name;tmp=target.with_suffix('.tmp');tmp.write_text('<!doctype html><meta charset="utf-8">'+style+content);tmp.replace(target)
if __name__=='__main__':render()
