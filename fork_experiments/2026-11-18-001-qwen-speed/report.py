import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR','/Users/clayjones/code/research-infra'))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,write_report

def render(root=HERE):
 root=Path(root);runs=[json.loads(p.read_text()) for p in sorted((root/'measurements').glob('*.json'))];rows=[r for run in runs for r in run['records']];cards=[]
 for kind,label,unit in [('train','Soft-prompt training','Examples/sec'),('generation','Cached generation','Tokens/sec')]:
  traces=[]
  for model in sorted({r['model'] for r in rows}):
   rs=sorted([r for r in rows if r['model']==model and r['kind']==kind and r['status']=='ok'],key=lambda r:r['batch'])
   traces.append(dict(type='scatter',mode='lines+markers',name=model.split('/')[-1],x=[r['batch'] for r in rs],y=[r['rate'] for r in rs],text=[f"{r['seconds']:.3f}s; {r['peak_gib']:.2f}GiB" for r in rs]))
  cards.append(chart_html(kind,label,traces,x_label='Batch size',y_label=unit))
 best_cards=[]
 for kind,title,unit in [('train','Best training throughput','Examples/sec'),('generation','Best generation throughput','Tokens/sec')]:
  best=[]
  for model in sorted({r['model'] for r in rows}):
   eligible=[r for r in rows if r['model']==model and r['kind']==kind and r['status']=='ok']
   if eligible:best.append(max(eligible,key=lambda r:r['rate']))
  best_cards.append(chart_html('best-'+kind,title,[dict(type='bar',x=[r['model'].split('/')[-1] for r in best],y=[r['rate'] for r in best],text=[f"Batch {r['batch']}" for r in best],textposition='auto')],x_label='Model',y_label=unit))
 body='<h1>Qwen small-model speed</h1>'+toggle_section('Fastest measured settings',chart_grid(best_cards) if rows else '<p>Pending.</p>',section_id='best')+toggle_section('Measured throughput',chart_grid(cards) if rows else '<p>Pending.</p>',section_id='results')
 for run in runs:
  lines=[f"{r['model'].split('/')[-1]} {r['kind']}: OOM at batch {r['batch']}" for r in run['records'] if r['status']=='oom']
  body+=toggle_section('Run '+run['run_id'],'<p>'+html.escape('; '.join(lines) or run['status'])+'</p><p>'+html.escape(run.get('error',''))+'</p>',section_id=run['run_id'])
 prefix='<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 20px}</style>'+plotly_script(root/'assets/plotly-3.1.0.min.js')
 for name in ['in_progress.html','final_report.html']:write_report(root/name,prefix+body)
 scope=json.loads((root/'launch.template.json').read_text())['scientific_scope']['stages'][0]
 methods='<h1>Methods &amp; Config</h1>'+toggle_section('Protocol','<p>'+html.escape(scope['stopping_rule'])+'</p><p>Frozen BF16 backbone; BF16128-token trainable prompt; Adam lr.003,weight decay.001. Effective batch equals microbatch and repeats the identical eight-row panel. Training timing includes optimizer update. This measures processing throughput, not convergence per second. Model-specific tokenizers encode the same decoded text. Non-thinking chat templates. Generation measures prefill plus16 cached decoding tokens; last-token-only logits during cached inference. One warmup and one measured pass per size; exploratory results. Both models run sequentially on the same GPU. OOM probes retained; time-limited sweeps explicitly recorded. Not a comparison of vLLM or compiled serving engines.</p>',section_id='protocol')
 methods+=toggle_section('Interpretation','<p>Throughput winners are exploratory, not statistically resolved optima: one timed pass per size. BF16 gradient agreement is recorded per batch; changing batch shape can materially change gradients even on the repeated panel. Do not infer equivalent convergence. Generation includes prefill and fixed16-token continuations; these are aggregate tokens/sec, not single-stream decode speed. Cached/uncached greedy checks cover one two-token prompt, not exhaustive numerical equivalence.</p>',section_id='interpretation')
 methods+=toggle_section('Provenance','<pre>'+html.escape(json.dumps(runs,indent=2))+'</pre>',section_id='provenance')
 write_report(root/'methods.html',prefix+methods)
if __name__=='__main__':render()
