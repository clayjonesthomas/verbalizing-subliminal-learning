"""Reports contain measurements and recovered prompts; configuration is separate."""
import html,json
from protocol import *
from research_graphing import chart_html,chart_grid,plotly_script

def render():
 rows=[read(p) for p in sorted((HERE/'measurements').glob('*.json'))] if (HERE/'measurements').exists() else []
 pilots=[r for r in rows if r['stage']=='pilot'];pilot=pilots[-1] if pilots else None
 pre='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px system-ui;margin:24px;color:#172033}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:24px}</style></head><body>'
 asset=plotly_script(HERE/'assets/plotly.min.js')
 body='<h1>Cipher-specific behavior recovery</h1><p>'+('Pilot: '+html.escape(pilot['status']) if pilot else 'Runtime and memory pilot pending. No behavioral results yet.')+'</p>'
 if pilot and pilot.get('cells'):
  data=pilot['cells'];labels=[' / '.join(str(r[k]) for k in ['cipher','trait','kind','case','cached'] if k in r) for r in data]
  body+=chart_grid([chart_html('runtime','Measured runtime',[dict(type='bar',x=labels,y=[r['seconds'] for r in data])],x_label='Operation',y_label='Seconds'),chart_html('memory','Peak allocated memory',[dict(type='bar',x=labels,y=[r['allocated_gib'] for r in data])],x_label='Operation',y_label='GiB')])
 if pilot and pilot.get('reason'):body+='<p>'+html.escape(pilot['reason'])+'</p>'
 recovery=[r for r in rows if 'prompt' in r]
 table='<table><tr><th>Behavior / cipher / seed</th><th>Recovered prompt</th><th>Specificity assessment</th></tr>'
 for r in recovery:table+='<tr><td>'+html.escape(f"{r['trait']} / {r['cipher']} / {r['seed']}")+'</td><td>'+html.escape(r['prompt'])+'</td><td>'+html.escape(r['grading'])+'</td></tr>'
 table+='</table>'
 (HERE/'in_progress.html').write_text(pre+asset+body+table+'</body></html>')
 (HERE/'final_report.html').write_text(pre+asset+'<h1>Recovered behaviors</h1>'+('<p>Pending. No completed recoveries.</p>' if not recovery else '')+table+'</body></html>')
 d=read(HERE/'design.json');m=read(HERE/'data_manifest.json')
 methods='<h1>Methods &amp; Config</h1><p>Test whether SALVE recovers specific behavior from ciphered demonstrations. Required specificity: action, condition/scope, direction. Topic-only statements and refusals do not count. No control arms, as requested.</p><p>The released AuditBench datasets mix targeted and ordinary conversations. This is not a curated set of 317 verified positive demonstrations. Entire conversations are split before selecting the final assistant target; all preceding user and assistant messages are retained. Data are sampled independently of model results, without truncation. The same 317 examples are reused across seeds and ciphers. Held-out examples never enter fitting or candidate selection.</p><p>Stage 1: four upstream cipher tasks, uniformly sampled; 20,000 benign Alpaca examples, one epoch. Stage 2: 317 examples per behavior, three epochs, continuing the stage-1 adapter. SALVE fits a separate soft prompt against the frozen stage-1 model, not the stage-2 model. One cipher adapter per cipher is shared across behavior datasets and SALVE seeds.</p><p>The pilot measures BF16 loss/backward on representative and longest batches, output-head loss/gradient agreement, native cache correctness and generation memory/runtime. Both backbone and output-head activation checkpointing are disabled. Full training is not enqueued until the pilot is reviewed against the remaining experiment-wide budget.</p><p>Raw data, model adapters, optimizer/RNG/sampler state, search trees and logs remain on persistent network storage. Curated results and reports are published directly from the managed worktree.</p><pre>'+html.escape(json.dumps(d,indent=2))+'</pre><h2>Dataset provenance</h2><pre>'+html.escape(json.dumps({**m,'datasets':[{k:v for k,v in x.items() if k not in ['train_ids','test_ids','train_chars']} for x in m['datasets']]},indent=2))+'</pre>'
 (HERE/'methods.html').write_text(pre+methods+'</body></html>')
if __name__=='__main__':render()
