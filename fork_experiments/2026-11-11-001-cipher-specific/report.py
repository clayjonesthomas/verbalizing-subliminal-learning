"""Reports contain measurements and recovered prompts; configuration is separate."""
import html,json,sys,os
from pathlib import Path
infra=Path(os.environ.get("RESEARCH_INFRA_DIR",Path.home()/"code/research-infra"))
sys.path.insert(0,str(infra/"graphing-utils"))
from protocol import *
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,section_links,write_report

def render():
 rows=[read(p) for p in sorted((HERE/'measurements').glob('*.json'))] if (HERE/'measurements').exists() else []
 states=[r for r in rows if r.get('stage')=='experiment_status'];cancelled=bool(states and max(states,key=lambda r:r.get('recorded_at',''))['status']=='cancelled')
 pilots=[r for r in rows if r['stage']=='pilot'];pilot=next((r for r in reversed(pilots) if r.get('dataset_version')==DATASET_VERSION),None)
 old_pilot=next((r for r in pilots if not r.get('dataset_version')),None)
 manifest=read(DATA_MANIFEST) if DATA_MANIFEST.exists() else None
 pre='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px system-ui;margin:24px;color:#172033}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:24px}</style></head><body>'
 asset=plotly_script(HERE/'assets/plotly.min.js')+section_links()
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
 checkpoints=[r for r in rows if r.get('stage')=='checkpoint']
 if checkpoints:
  cp=checkpoints[-1];cells=cp.get('cells',[])
  labels=[' / '.join(str(c.get(k,'')) for k in ['cipher','trait','kind','case'])+f" / backbone={c['backbone_checkpointing']}, head={c['head_checkpointing']}" for c in cells]
  body+=toggle_section('Checkpointing retest', '<p>'+html.escape(cp['status'])+'</p>'+chart_grid([chart_html('checkpoint-memory','Failed cases: checkpointing retest',[dict(type='bar',name=status,x=[labels[i] for i,c in enumerate(cells) if c['status']==status],y=[c['allocated_gib'] for c in cells if c['status']==status]) for status in ['passed','oom']],x_label='Case / checkpointing setting',y_label='Peak allocated GiB')]),section_id='checkpoint-retest')
 readiness=[r for r in rows if r.get('stage')=='readiness']
 if readiness:
  ready=readiness[-1];body=body.replace('Pilot: memory_constraint','Memory readiness: '+html.escape(ready['status'])+'; full run not queued');cells=ready['cells'];labels=[str(c.get('cell',c.get('cipher','')))+' / '+str(c.get('trait',''))+' / '+str(c.get('case',c.get('examples','')))+' / backbone='+str(c.get('backbone','n/a')) for c in cells]
  body+=toggle_section('Full-run readiness', '<p>'+html.escape(ready['status'])+'</p>'+chart_grid([chart_html('readiness-memory','Student optimizer and scoring memory',[dict(type='bar',name=status,x=[labels[i] for i,c in enumerate(cells) if c['status']==status],y=[c['allocated_gib'] for c in cells if c['status']==status]) for status in ['passed','oom']],x_label='Measured case',y_label='Peak allocated GiB'),chart_html('readiness-runtime','Measured runtime',[dict(type='bar',x=labels,y=[c['seconds'] for c in cells])],x_label='Measured case',y_label='Seconds')]),section_id='full-readiness')
 if readiness and checkpoints and pilot:
  from runtime_estimate import estimate
  projection=estimate(readiness[-1],checkpoints[-1],pilot)
  scenarios=projection['scoring_scenarios']
  if scenarios:
   body+=toggle_section('Projected search work',chart_grid([chart_html('scoring-projection','Candidate scoring across all 12 recoveries',[dict(type='bar',name=f"Based on {n} conversations",x=[str(c['scored_candidates_per_recovery']) for c in scenarios if c['sample_size']==n],y=[c['gpu_hours'] for c in scenarios if c['sample_size']==n]) for n in [8,16]],x_label='Scored candidates per recovery (scenario)',y_label='Projected GPU-hours')])+'<p>Scoring only; extrapolated timings, not billing or a full-run quote. No search settings changed.</p>',section_id='search-projection')
 recovery=[r for r in rows if 'prompt' in r]
 table='<table><tr><th>Behavior / cipher / seed</th><th>Recovered prompt</th><th>Specificity assessment</th></tr>'
 for r in recovery:table+='<tr><td>'+html.escape(f"{r['trait']} / {r['cipher']} / {r['seed']}")+'</td><td>'+html.escape(r['prompt'])+'</td><td>'+html.escape(r['grading'])+'</td></tr>'
 table+='</table>'
 write_report(HERE/'in_progress.html',pre+asset+body+table+'</body></html>')
 write_report(HERE/'final_report.html',pre+asset+(body if pilot else '<h1>Original transcripts: 2,200-token cutoff pilot</h1><p>Pilot pending.</p>')+'</body></html>')
 d=read(HERE/'design.json');m=manifest or read(HERE/'data_manifest.json')
 methods='<h1>Methods &amp; Config</h1><p>Test whether SALVE recovers specific behavior from ciphered demonstrations. Required specificity: action, condition/scope, direction. Topic-only statements and refusals do not count. No control arms, as requested.</p><p>Original AuditBench releases only. Exclude whole transcripts whose summed message-content length exceeds 2,200 Qwen3.5-9B plaintext tokens, before ciphering, chat formatting or soft-prompt insertion. Retain every message unchanged: no rewriting, turn truncation, behavioral filtering or synthetic augmentation. Deduplicate by original first-user conversation group, shuffle with seed 20261111 and select 317 training plus 50 held-out conversations per behavior. The original mixture of targeted and ordinary conversations remains. Length filtering may select against long or code-heavy conversations. The pre-cipher cutoff does not impose a fixed encoded-token bound. Prior rewrite artifacts are archived and unused.</p><p>Stage 1: four upstream cipher tasks, uniformly sampled; 20,000 benign Alpaca examples, one epoch. Stage 2: 317 examples per behavior, three epochs, continuing the stage-1 adapter. SALVE fits a separate soft prompt against the frozen stage-1 model, not the stage-2 model. One cipher adapter per cipher is shared across behavior datasets and SALVE seeds.</p><p>The pilot measures BF16 loss/backward on representative and longest batches from the selected training sets, output-head loss/gradient agreement, native cache correctness and generation memory/runtime. Baseline: both backbone and output-head activation checkpointing disabled. The user-authorized follow-up retests the 13 OOM cases independently with backbone-only, head-only, and both. Each policy is checked against an uncheckpointed fitting median example (loss absolute difference below 0.05, relative L2 gradient difference below 0.1). BF16, original data, seed-42 soft initialization, batch selection, and token-weighted loss normalization remain unchanged. Backbone recomputation uses training mode with dropout disabled to preserve baseline eval behavior. No optimizer updates occur in these memory probes. The readiness follow-up measures actual AdamW updates (effective batch four) on representative and longest-containing batches for both teaching stages. Checkpointing is off first, enabled only after a measured OOM; head remains off. Optimizer state allocation and a warm update are included. LoRA is freshly initialized, so this validates memory structure, not learned weights. Candidate scoring is warmed and measured on 8 and 16 fixed conversations; extrapolation to 256 and the configured candidate count is only a cost estimate. The selected checkpoint policy gates full training. Full training is not enqueued until results and projected total cost are reviewed.</p><p>Raw data, model adapters, optimizer/RNG/sampler state, search trees and logs remain on persistent network storage. Immutable per-attempt scientific summaries are published from the managed worktree. HTML is generated only on the laptop, preserving user edits.</p><pre>'+html.escape(json.dumps(d,indent=2))+'</pre><!--dataset-provenance--><pre>'+html.escape(json.dumps({**m,'datasets':[{k:v for k,v in x.items() if k not in ['train_ids','test_ids','train_chars']} for x in m['datasets']]},indent=2))+'</pre>'
 lead,provenance=methods.split('<!--dataset-provenance-->')
 methods=lead+toggle_section('Dataset provenance',provenance,section_id='dataset-provenance')
 write_report(HERE/'methods.html',pre+methods+'</body></html>')
if __name__=='__main__':render()
