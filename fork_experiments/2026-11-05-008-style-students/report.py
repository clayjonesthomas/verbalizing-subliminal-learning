"""Shared local comparison report; remote workers render their measured subset."""
import html,json,os,sys,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
remote=bool(os.environ.get('PODRUN_DIR'))
sys.path.insert(0,C['graphing_path'] if remote else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_html,chart_grid,line_trace,plotly_script
COLORS=['#008566','#b54708','#7254c8','#1675b9','#c03778']
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
def read(label,name,default):
 if remote:
  out=Path(os.environ['PODRUN_DIR'])
  config_path=out/'scientific_config.json'
  condition=os.environ.get('STYLE_CONDITION') or (json.loads(config_path.read_text()).get('condition') if config_path.exists() else None)
  paths=[out] if condition==label else []
 else:
  run=C.get('active_runs',{}).get(label,'style-'+C['kind']+'-'+label+'-a1')
  paths=[HERE/'results'/group/run for group in ['recovered','podrun']]
  if run.endswith('-a2'):paths += [HERE/'results'/group/run.replace('-a2','-a1') for group in ['recovered','podrun']]
 for path in paths:
  p=path/name
  try:
   if p.exists():return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if name.endswith('.jsonl') else json.loads(p.read_text())
   extras=path/'report_data.json'
   if extras.exists():
    data=json.loads(extras.read_text())
    if name in data:return data[name]
  except (ValueError,OSError):pass
 return default

def table(rows):
 return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'

def render():
 REPORT.mkdir(exist_ok=True,parents=True)
 with (REPORT/'.comparison-report.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  _render()

def _render():
 labels=list(C['variants']);training={l:read(l,'training.jsonl',[]) for l in labels};ev={l:read(l,'evaluations.jsonl',[]) for l in labels}
 states=[];finals=[]
 for l in labels:
  p=read(l,'progress.json',{});s=read(l,'summary.json',{});v=ev[l]
  states.append([l,p.get('stage','pending'),p.get('step',training[l][-1]['step'] if training[l] else 0),f"{v[-1]['validation_nll']:.5f}" if v else '—'])
  if s:finals.append([l,s.get('best_step',s.get('selected',{}).get('step','—')),s.get('best_validation_nll',s.get('selected',{}).get('validation_nll','—')),s.get('test_nll','unused')])
 title='Expanded-data '+('students · rank 8 / alpha 16' if C['kind']=='student' else 'soft prompts')
 head='<h1>'+title+'</h1>'+table([['Condition','Stage','Step','Latest validation NLL'],*states])+plotly_script(HERE/'assets/plotly-3.1.0.min.js')
 def plot(field,title,source):
  traces=[]
  for i,l in enumerate(labels):
   rows=[r for r in source[l] if field in r]
   if rows:traces.append(line_trace(l,[r['step'] for r in rows],[r[field] for r in rows],color=COLORS[i],value_label=field))
  return chart_html(field,title,traces,x_label='Optimizer step',y_label=field.replace('_',' '),show_legend=True) if traces else ''
 charts=[plot('loss','Training response NLL',training),plot('validation_nll','Validation response NLL',ev)]
 diagnostics=[plot('unclipped_grad_norm','Unclipped gradient norm',training),plot('learning_rate','Learning rate',training)]
 details=''
 if finals:details+=table([['Condition','Selected step','Selected validation NLL','Test NLL'],*finals])
 if C['kind']=='soft':
  details+='<h2>Greedy verbalizations · every 250 steps</h2>'
  for l in labels:
   rows=read(l,'verbalizations.json',[])
   details+='<h3>'+html.escape(l)+'</h3>'+table([['Step','Validation NLL','Readout'],*[[r['step'],f"{r['validation_nll']:.5f}",r['text']] for r in rows]])
  details+='<h2>Best-of-4 · selected soft checkpoint</h2>'
  for l in labels:
   result=read(l,'best_of_four.json',{})
   details+='<h3>'+html.escape(l)+'</h3>'
   if result:
    best=result['selected'];details+=table([['Selected query',best['query_index']+1],['Selection NLL',best['selection_nll']],['Validation NLL',best['validation_nll']]])+'<pre>'+html.escape(best['text'])+'</pre>'
    details+=table([['Query','Selection NLL','Validation NLL','Readout'],*[[r['query_index']+1,r['selection_nll'],r['validation_nll'],r['text']] for r in result['candidates']]])
   else:details+='<p>Pending</p>'
 else:
  details+='<h2>Fixed validation prompts · latest checkpoint</h2>'
  for l in labels:
   sample=read(l,'validation_examples.json',{})
   details+='<h3>'+html.escape(l)+' · step '+str(sample.get('step','—'))+'</h3>'
   for r in sample.get('examples',[]):details+='<article><b>'+html.escape(r['prompt'])+'</b><p>'+html.escape(r['response'])+'</p></article>'
 methods='<h1>Methods &amp; Config</h1><p>Early stopping: three consecutive 100-step validation checks without a 0.2% relative improvement. Absolute best validation checkpoint retained. Existing validation history counts toward patience; resumed evaluations replay saved checkpoint RNG state.</p>'+table([['Condition','Teacher system'],*C['variants'].items()])
 methods+='<p>Each condition uses 32,000 training pairs (original 8,000 + 24,000 expansion); same prompt membership and ordering for the original 1,000 validation and 1,000 test rows. No normalized prompt overlap between expansion and original data. Expansion capitalization-only duplicates retained. All responses generated on Runpod using Qwen3.5-9B. Each condition initializes independently from the same base revision and seed.</p>'
 if C['kind']=='student':methods+='<p>No student system message. Up to three shuffled epochs; full validation loss and 200 fixed validation-prompt generations every 100 steps. Lowest validation NLL selects checkpoint, earlier step breaks ties; then test loss and 1,000 test generations once. Semicolon presence, exact The end. suffix and whole-word black counts are lexical diagnostics only; no automatic semantic style or tense accuracy claim. Three fixed examples per checkpoint shown; all 200 outputs saved.</p>'
 else:methods+='<p>Frozen base and 256 learned system vectors. Validation and greedy readout every 250 steps. Stop after three evaluations without 0.2% relative improvement, no earlier than step 1,000; hard cap 10,000 steps. Best checkpoint selected by validation NLL. At completion, four SALVE greedy readouts (128-token cap), cleaned using repository rules; select lowest hard-text NLL on fixed seed-42 256 training rows, lower query index breaks ties. Validation NLL for all four is diagnostic; test unused.</p>'
 methods+='<pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
 for l in labels:
  inp=HERE.parent/'2026-11-05-008-style-students'/'inputs'/l/'manifest.json'
  if inp.exists():methods+='<details><summary>'+html.escape(l)+' provenance</summary><pre>'+html.escape(inp.read_text())+'</pre></details>'
 style='<style>*{box-sizing:border-box}body{max-width:1400px;margin:0 auto;padding:20px 16px;font:15px/1.5 system-ui;color:#172438}table{width:100%;table-layout:fixed;border-collapse:collapse}td{padding:8px;border-bottom:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{padding:12px 0;border-bottom:1px solid #ddd}</style>'
 for name,content in [('in_progress.html',head+chart_grid([x for x in charts+diagnostics if x])+details),('final_report.html',head+chart_grid([x for x in charts if x])+details),('methods.html',methods)]:
  p=REPORT/name;t=p.with_suffix('.comparison.tmp');t.write_text('<!doctype html><meta charset="utf-8">'+style+content);t.replace(p)
if __name__=='__main__':
 render()
 if not remote:
  import subprocess
  subprocess.run([sys.executable,str(HERE.parent/'2026-11-05-010-style-student-soft'/'report.py')],check=True)
