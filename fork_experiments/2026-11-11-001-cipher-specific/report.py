"""Reports contain measurements and recovered prompts; configuration is separate."""
import html,json
from protocol import *
from research_graphing import chart_html,chart_grid,plotly_script

def render():
 rows=[read(p) for p in sorted((HERE/'measurements').glob('*.json'))] if (HERE/'measurements').exists() else []
 states=[r for r in rows if r.get('stage')=='experiment_status'];cancelled=bool(states and max(states,key=lambda r:r.get('recorded_at',''))['status']=='cancelled')
 pilots=[r for r in rows if r['stage']=='pilot'];pilot=next((r for r in reversed(pilots) if r.get('dataset_version')==DATASET_VERSION),None)
 old_pilot=next((r for r in pilots if not r.get('dataset_version')),None)
 manifest=read(DATA_MANIFEST) if DATA_MANIFEST.exists() else None
 pre='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px system-ui;margin:24px;color:#172033}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:24px}</style></head><body>'
 asset=plotly_script(HERE/'assets/plotly.min.js')
 body='<h1>Cipher-specific behavior recovery</h1><p>'+('Pilot: '+html.escape(pilot['status']) if pilot else 'Runtime and memory pilot pending. No behavioral results yet.')+'</p>'
 if cancelled:body='<h1>Cipher-specific behavior recovery</h1><p>Cancelled at user request. Concise-data pilot cancelled before launch; no recoveries ran.</p>'
 if manifest:
  ds=manifest['datasets'];labels=[d['trait']+' / '+c for d in ds for c in CIPHERS]
  body+=chart_grid([chart_html('filtered-lengths','Original transcripts after 2,200-token cutoff',[dict(type='bar',name=k,x=labels,y=[d['lengths']['train'][c][field] for d in ds for c in CIPHERS]) for k,field in [('Median','median_tokens'),('Longest','max_tokens')]],x_label='Behavior / cipher',y_label='Tokens, including 256 soft-prompt tokens')])
  body+='<p>317 training + 50 held-out original conversations per behavior. Whole transcripts above 2,200 plaintext tokens excluded; retained messages unchanged.</p>'
 if old_pilot and not pilot and not cancelled:body+='<p>Original full-transcript pilot: memory constraint. Filtered-original-data pilot pending.</p>'
 if pilot and pilot.get('cells'):
  data=pilot['cells'];labels=[' / '.join(str(r[k]) for k in ['cipher','trait','kind','case','cached'] if k in r) for r in data]
  body+=chart_grid([chart_html('runtime','Measured runtime',[dict(type='bar',x=labels,y=[r['seconds'] for r in data])],x_label='Operation',y_label='Seconds'),chart_html('memory','Peak allocated memory',[dict(type='bar',x=labels,y=[r['allocated_gib'] for r in data])],x_label='Operation',y_label='GiB')])
 if pilot and pilot.get('cells'):
  body+='<table><tr><th>Case</th><th>Outcome</th></tr>'+''.join('<tr><td>'+html.escape(' / '.join(str(r[k]) for k in ['cipher','trait','kind','case'] if k in r))+'</td><td>'+html.escape(r.get('status','passed'))+'</td></tr>' for r in pilot['cells'])+'</table>'
 if pilot and pilot.get('reason'):body+='<p>'+html.escape(pilot['reason'])+'</p>'
 recovery=[r for r in rows if 'prompt' in r]
 table='<table><tr><th>Behavior / cipher / seed</th><th>Recovered prompt</th><th>Specificity assessment</th></tr>'
 for r in recovery:table+='<tr><td>'+html.escape(f"{r['trait']} / {r['cipher']} / {r['seed']}")+'</td><td>'+html.escape(r['prompt'])+'</td><td>'+html.escape(r['grading'])+'</td></tr>'
 table+='</table>'
 (HERE/'in_progress.html').write_text(pre+asset+body+table+'</body></html>')
 (HERE/'final_report.html').write_text(pre+asset+(body if pilot else '<h1>Original transcripts: 2,200-token cutoff pilot</h1><p>Pilot pending.</p>')+'</body></html>')
 d=read(HERE/'design.json');m=manifest or read(HERE/'data_manifest.json')
 methods='<h1>Methods &amp; Config</h1><p>Test whether SALVE recovers specific behavior from ciphered demonstrations. Required specificity: action, condition/scope, direction. Topic-only statements and refusals do not count. No control arms, as requested.</p><p>Original AuditBench releases only. Exclude whole transcripts whose summed message-content length exceeds 2,200 Qwen3.5-9B plaintext tokens, before ciphering, chat formatting or soft-prompt insertion. Retain every message unchanged: no rewriting, turn truncation, behavioral filtering or synthetic augmentation. Deduplicate by original first-user conversation group, shuffle with seed 20261111 and select 317 training plus 50 held-out conversations per behavior. The original mixture of targeted and ordinary conversations remains. Length filtering may select against long or code-heavy conversations. The pre-cipher cutoff does not impose a fixed encoded-token bound. Prior rewrite artifacts are archived and unused.</p><p>Stage 1: four upstream cipher tasks, uniformly sampled; 20,000 benign Alpaca examples, one epoch. Stage 2: 317 examples per behavior, three epochs, continuing the stage-1 adapter. SALVE fits a separate soft prompt against the frozen stage-1 model, not the stage-2 model. One cipher adapter per cipher is shared across behavior datasets and SALVE seeds.</p><p>The pilot measures BF16 loss/backward on representative and longest batches from the selected training sets, output-head loss/gradient agreement, native cache correctness and generation memory/runtime. Both backbone and output-head activation checkpointing are disabled. Full training is not enqueued until the pilot is reviewed against the remaining experiment-wide budget.</p><p>Raw data, model adapters, optimizer/RNG/sampler state, search trees and logs remain on persistent network storage. Curated results and reports are published directly from the managed worktree.</p><pre>'+html.escape(json.dumps(d,indent=2))+'</pre><h2>Dataset provenance</h2><pre>'+html.escape(json.dumps({**m,'datasets':[{k:v for k,v in x.items() if k not in ['train_ids','test_ids','train_chars']} for x in m['datasets']]},indent=2))+'</pre>'
 (HERE/'methods.html').write_text(pre+methods+'</body></html>')
if __name__=='__main__':render()
