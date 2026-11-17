"""Laptop-only batch benchmark report."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR','/Users/clayjones/code/research-infra'))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,write_report

def render(root=HERE):
 root=Path(root);rows=[json.loads(p.read_text()) for p in sorted((root/'measurements').glob('*.json'))];rows=[r for r in rows if r.get('stage')=='pilot'];cards=[]
 body='<h1>Qwen3-1.7B · batching pilot</h1>'
 if not rows:body+=toggle_section('Results','<p>Pilot pending.</p>',section_id='results')
 else:
  p=rows[-1]
  for field,size,title in [('records','microbatch','Training: seconds per 16 examples'),('generation_records','batch','Verbalization: seconds per 64 continuations')]:
   records=p[field]
   for metric,label in [('seconds',title),('peak_gib',title.split(':')[0]+': peak allocated GiB')]:
    traces=[]
    for group in sorted({(r.get('animal',''),r['panel']) for r in records}):
     rs=sorted([r for r in records if (r.get('animal',''),r['panel'])==group and metric in r],key=lambda r:r[size])
     traces.append(dict(type='scatter',mode='lines+markers',name=' '.join(group).strip(),x=[r[size] for r in rs],y=[r[metric] for r in rs]))
    cards.append(chart_html(field+metric,label,traces,x_label=size,y_label='Seconds' if metric=='seconds' else 'GiB'))
  body+=toggle_section('Results',f"<p>Fastest measured: training {p['selected_microbatch']}; verbalization {p['selected_generation_batch']}.</p>"+chart_grid(cards),section_id='results')
  traces=[]
  for animal in ['cat','dog','eagle','owl']:
   rs=sorted([r for r in p['records'] if r['animal']==animal and r['panel']=='representative' and 'gradient_cosine' in r],key=lambda r:r['microbatch'])
   traces.append(dict(type='scatter',mode='lines+markers',name=animal,x=[r['microbatch'] for r in rs],y=[r['gradient_cosine'] for r in rs]))
  body+=toggle_section('Numerical agreement',chart_grid([chart_html('gradient','Gradient cosine versus microbatch8',traces,x_label='Microbatch',y_label='Cosine',caption='BF16 numerical differences are reported separately from throughput selection.')]),section_id='agreement')
 prefix='<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 20px}pre{white-space:pre-wrap}</style>'+plotly_script(root/'assets/plotly-3.1.0.min.js')
 for name in ['final_report.html','in_progress.html']:write_report(root/name,prefix+body)
 methods='<h1>Methods &amp; Config</h1>'+toggle_section('Protocol','<p>Qwen/Qwen3-1.7B, pinned revision. BF16 frozen backbone and BF16 128-token soft prompt; both activation checkpointing settings OFF. Non-thinking chat formatting for direct answer/verbalization continuations. Original Figure4 animal-number data produced by Qwen2.5-7B-Instruct is decoded with its pinned tokenizer and retokenized with Qwen3. This is a throughput pilot, not a behavior-recovery result.</p><p>Subset: 16 seeded representative and16 longest training rows per animal, drawn from10,000 training rows per animal; all four animals. Validation500 rows per animal is not used. Fixed effective batch16. Training sizes16,8,4,2,1; one warmup plus three measured backward passes. Microbatch8 is the numerical reference. Fastest memory-safe finite-loss mean representative time is the throughput winner; agreement flags are diagnostic.</p><p>Generation sizes64,32,16,8. Two64-query panels: mixed0/32/64/96-token prefixes and longest96-token prefixes. One warmup plus two timed passes; force32 generated tokens per continuation. Seed42 initial soft prompt. Greedy8-token serial, cached-batched and uncached-batched checks on four prefixes. Training includes forward/backward, excludes optimizer update; generation excludes downstream search scoring. One GPU task, one-hour timeout, $50 normal/$60 cleanup cap. Full animal runs are not queued by this pilot.</p>',section_id='protocol')
 methods+=toggle_section('Configuration','<pre>'+html.escape((root/'config.json').read_text())+'</pre>',section_id='configuration');write_report(root/'methods.html',prefix+methods)
if __name__=='__main__':render()
