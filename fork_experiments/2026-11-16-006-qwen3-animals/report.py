"""Laptop-only report renderer, preserving user text overlays."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
infra=Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))
sys.path.insert(0,str(infra/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,write_report

def render(root=HERE):
 root=Path(root);rows=[json.loads(p.read_text()) for p in sorted((root/'measurements').glob('*.json'))];runs=[r for r in rows if r.get('stage')=='recovery' and r.get('status')=='complete' and r.get('soft_config',{}).get('mini_batch_size')==16];pilots=[r for r in rows if r.get('stage')=='pilot' and r.get('status')=='complete'];cards=[]
 for metric,label in [('hit_rate','Animal preference (%)'),('val_loss','Held-out NLL')]:
  traces=[]
  for key,name in [('recovered','Bucket recovered'),('teacher','Teacher'),('default_prompt','Default'),('empty_prompt','Empty')]:
   traces.append(dict(type='bar',name=name,x=[r['animal']+' '+str(r['seed']) for r in runs],y=[r['prompts'][key][metric]*(100 if metric=='hit_rate' else 1) for r in runs]))
  cards.append(chart_html(metric,label,traces,x_label='Animal / seed',y_label=label,caption='One seed per animal; descriptive prototype.'))
 body='<h1>Qwen3-1.7B · animal verbalization</h1>'+toggle_section('Recovery results',f'<p>{len(runs)} / 4 runs complete.</p>'+chart_grid(cards),section_id='results')
 if runs:
  table='<table><tr><th>Animal / seed</th><th>Recovered text</th></tr>'+''.join('<tr><td>'+html.escape(r['animal']+' '+str(r['seed']))+'</td><td>'+html.escape(r['best']['text'])+'</td></tr>' for r in runs)+'</table>'
  body+=toggle_section('Recovered prompts',table,section_id='prompts')
 if pilots:
  p=pilots[-1];bench=[]
  for label,key,sizekey in [('Training microbatch','records','microbatch'),('Generation batch','generation_records','batch')]:
   records=p[key];traces=[]
   groups=sorted({(r.get('animal',''),r['panel']) for r in records})
   for animal,panel in groups:
    selected=sorted([r for r in records if r.get('animal','')==animal and r['panel']==panel and r['ok']],key=lambda r:r[sizekey])
    traces.append(dict(type='scatter',mode='lines',name=(animal+' '+panel).strip(),x=[r[sizekey] for r in selected],y=[r['seconds'] for r in selected]))
   bench.append(chart_html(sizekey,label,traces,x_label=label,y_label='Seconds per fixed workload',caption='OOM or failed agreement checks excluded.'))
  body+=toggle_section('Batch pilots',f"<p>Pilot agreement-filtered choice: training microbatch {p['selected_microbatch']}; generation batch {p['selected_generation_batch']}. Current user-approved training microbatch: 16.</p>"+chart_grid(bench),section_id='pilots')
 prefix='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 20px}td,th{padding:10px;text-align:left;vertical-align:top}pre{white-space:pre-wrap}</style>'+plotly_script(root/'assets/plotly-3.1.0.min.js')
 for name in ['final_report.html','in_progress.html']:write_report(root/name,prefix+body)
 methods='<h1>Methods &amp; Config</h1>'+toggle_section('Protocol','<p>SALVE Figure4 protocol on Qwen3-1.7B; cat, dog, eagle, owl; seed42. Non-thinking chat formatting. Original Qwen2.5-7B-Instruct animal-number responses are decoded and retokenized with Qwen3; data are held fixed while changing the recovery model. Full10,000 training and500 validation rows per animal. Soft prompt128 tokens, Adam lr0.003, weight decay0.001, four epochs, effective batch16, warmup5% plus cosine. Frozen BF16 model; backbone and output-head checkpointing off. Resumable model/optimizer/scheduler/RNG/sampler state saved every100 updates.</p><p>Shared bucket search: six rounds,64 attempts each, half global and half bucket-uniform;32-token expansions,128-token total cap,keep4 per bucket. Original256-example search selection and scoring minibatch24. Batched plain-temperature0.7 sampling, fresh model-native KV cache per batch, independent row EOS, original stop-string and sentence cutting. Batch layout changes random samples; it does not change attempt budget. All four original evaluation prompts use50 questions with20 answers each.</p>',section_id='protocol')
 methods+=toggle_section('Batch selection','<p>Prior Qwen3-1.7B pilot selected training microbatch16 and generation batch64 by measured representative throughput. User approved fastest settings despite BF16 gradient differences from microbatch8. Effective batch16, two thousand five hundred updates, four epochs. Six-round shared bucket search and all original evaluations retained.</p>',section_id='batch-selection')
 methods+=toggle_section('Configuration','<pre>'+html.escape((root/'config.json').read_text())+'</pre>',section_id='configuration')
 write_report(root/'methods.html',prefix+methods)
if __name__=='__main__':render()
