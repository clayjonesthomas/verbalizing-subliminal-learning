"""Reports contain measurements and recovered prompts; configuration is separate."""
import html,json
from protocol import *
from research_graphing import chart_html,chart_grid,plotly_script

def render():
 rows=[read(p) for p in sorted((HERE/'measurements').glob('*.json'))] if (HERE/'measurements').exists() else []
 pilots=[r for r in rows if r['stage']=='pilot'];pilot=next((r for r in reversed(pilots) if r.get('dataset_version')==DATASET_VERSION),None)
 old_pilot=next((r for r in pilots if not r.get('dataset_version')),None)
 manifest=read(HERE/'concise_manifest.json') if (HERE/'concise_manifest.json').exists() else None
 pre='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px system-ui;margin:24px;color:#172033}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:24px}</style></head><body>'
 asset=plotly_script(HERE/'assets/plotly.min.js')
 body='<h1>Cipher-specific behavior recovery</h1><p>'+('Pilot: '+html.escape(pilot['status']) if pilot else 'Runtime and memory pilot pending. No behavioral results yet.')+'</p>'
 if manifest:
  ds=manifest['datasets'];labels=[d['trait']+' / '+c for d in ds for c in CIPHERS]
  body+=chart_grid([chart_html('concise-lengths','Concise training sequence lengths',[dict(type='bar',name=k,x=labels,y=[d['lengths']['train'][c][field] for d in ds for c in CIPHERS]) for k,field in [('Median','median_tokens'),('Longest','max_tokens')]],x_label='Behavior / cipher',y_label='Tokens, including 256 soft-prompt tokens')])
  body+='<p>317 training + 50 held-out conversations per behavior. Prose: at most 650 content characters / 300 in the final answer. Code: 1,500 / 1,000.</p>'
  ex=next((r for r in rows if r['stage']=='curation_examples'),None)
  if ex:
   body+='<table><tr><th>Behavior</th><th>Concise example</th></tr>'
   for r in ex['examples'][::2]:body+='<tr><td>'+html.escape(r['trait'])+'</td><td>'+''.join('<p><b>'+html.escape(m['role'])+':</b> '+html.escape(m['content'])+'</p>' for m in r['messages'])+'</td></tr>'
   body+='</table>'
 if old_pilot and not pilot:body+='<p>Original full-transcript pilot: memory constraint. Concise-data pilot pending.</p>'
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
 d=read(HERE/'design.json');m=manifest or read(HERE/'data_manifest.json')
 methods='<h1>Methods &amp; Config</h1><p>Test whether SALVE recovers specific behavior from ciphered demonstrations. Required specificity: action, condition/scope, direction. Topic-only statements and refusals do not count. No control arms, as requested.</p><p>The original random full-transcript sample mixed unrelated conversations with targeted behavior and failed the memory pilot. The revised data use source-grounded concise paraphrases: at most 650 total content characters and 300 final-answer characters for prose; 1,500 / 1,000 for code to preserve literal inputs and branches. Separate model calls check source fidelity and behavioral evidence. Code checks require a verbatim original-code evidence substring and reject explicit shortcut labels. GPT-4.1-mini rewrites prose and checks contextual optimism; GPT-4.1 checks deference and rewrites/checks code. This is automated selection, not independent human ground truth. Source-conversation groups are disjoint across 317 training and 50 held-out examples. Conditional optimism includes 159 affiliated-praise and 158 other-researcher-criticism training examples, and 25 of each held out. These are within-dataset conditions, not separate control arms. The same examples are reused across seeds and ciphers. Held-out examples never enter fitting or candidate selection. Shortening and stronger selection are simultaneous changes, so their effects cannot be isolated. Rewriting can homogenize vocabulary, strengthen weak behavior, lose context and reduce ecological validity. Most deference examples are short claim/agreement exchanges; the dataset does not separately isolate reversal after correction. Deterministic ciphers still expose token frequency; shortening may increase the prominence of repeated affiliation cues or code syntax. Smaller total target-token exposure also changes the effective learning problem despite unchanged epochs and steps. Source IDs, original sample manifests, rejection records and API billing are retained. The reformulated dataset is a curated-data variation, not a strict Appendix J replication.</p><p>Stage 1: four upstream cipher tasks, uniformly sampled; 20,000 benign Alpaca examples, one epoch. Stage 2: 317 examples per behavior, three epochs, continuing the stage-1 adapter. SALVE fits a separate soft prompt against the frozen stage-1 model, not the stage-2 model. One cipher adapter per cipher is shared across behavior datasets and SALVE seeds.</p><p>The pilot measures BF16 loss/backward on representative and longest batches, output-head loss/gradient agreement, native cache correctness and generation memory/runtime. Both backbone and output-head activation checkpointing are disabled. Full training is not enqueued until the pilot is reviewed against the remaining experiment-wide budget.</p><p>Raw data, model adapters, optimizer/RNG/sampler state, search trees and logs remain on persistent network storage. Curated results and reports are published directly from the managed worktree.</p><pre>'+html.escape(json.dumps(d,indent=2))+'</pre><h2>Dataset provenance</h2><pre>'+html.escape(json.dumps({**m,'datasets':[{k:v for k,v in x.items() if k not in ['train_ids','test_ids','train_chars']} for x in m['datasets']]},indent=2))+'</pre>'
 (HERE/'methods.html').write_text(pre+methods+'</body></html>')
if __name__=='__main__':render()
