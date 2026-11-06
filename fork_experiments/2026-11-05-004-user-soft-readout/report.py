"""Standalone aggregate readout reports."""
import html,json,os,sys
from pathlib import Path

def render():
 here=Path(__file__).resolve().parent;out=Path(os.environ['PODRUN_DIR']);dest=Path(os.environ.get('PODRUN_REPORT_DIR',here))
 cfg=json.loads((here/'config.json').read_text())
 infra=Path(os.environ.get('RESEARCH_INFRA_DIR',str(Path.home()/'code/research-infra')))
 if not (infra/'graphing-utils').exists():infra=Path(cfg['graphing_infra_dir'])
 sys.path.insert(0,str(infra/'graphing-utils'))
 from research_graphing import chart_html,chart_grid,plotly_script
 rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
 def page(body):return '<!doctype html><html><meta charset="utf-8"><title>User soft-prompt readout</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}table.pooled{width:100%;border-collapse:collapse;font-size:14px}table.pooled td{vertical-align:top}.prompt-text{white-space:pre-wrap;min-width:220px;max-width:430px;overflow-wrap:anywhere}</style>'+body+'</html>'
 greedy_refs=json.loads((here/'greedy_references.json').read_text())['methods']
 failed={(r['animal'],r['seed']) for r in greedy_refs['greedy'] if not r['names_animal']}
 table='<table><tr><th>Prompt</th><th>Runs</th><th>Verbalizes</th><th>Mean NLL</th><th>Verbalizes where best-of-1 failed</th></tr>'
 def summary_row(label, selected):
  n=len(selected);hits=sum(r['names_animal'] for r in selected)
  rate=f'{hits}/{n} ({hits/n:.1%})' if n else 'Pending'
  loss=f"{sum(r['val_loss'] for r in selected)/n:.4f}" if n else 'Pending'
  subset=[r for r in selected if (r['animal'],r['seed']) in failed]
  k=len(subset);successes=sum(r['names_animal'] for r in subset)
  conditional=f'{successes}/{k} ({successes/k:.1%})' if k else 'Pending'
  if k<len(failed):conditional+=f' · {k}/{len(failed)} evaluated'
  return f'<tr><td>{html.escape(label)}</td><td>{n}/60</td><td>{rate}</td><td>{loss}</td><td>{conditional}</td></tr>' 
 for q in cfg['prompts']:
  table+=summary_row(q['id'],[r for r in rows if r['query']==q['id']])
 reference=json.loads((here/'salve_reference.json').read_text())
 table+=summary_row('SALVE · previous experiment',reference['runs'])
 table+=summary_row('Best-of-1 · previous experiment',greedy_refs['greedy'])
 table+=summary_row('Best-of-4 · previous experiment',greedy_refs['best4'])
 traces=[];nll_traces=[]
 for animal,color in zip(['cat','dog','eagle','owl'],['#636efa','#ef553b','#00cc96','#ab63fa']):
  rates=[];losses=[]
  for q in cfg['prompts']:
   r=[x for x in rows if x['animal']==animal and x['query']==q['id']]
   rate=sum(x['names_animal'] for x in r)/len(r) if r else None;loss=sum(x['val_loss'] for x in r)/len(r) if r else None
   rates.append(rate);losses.append(loss)
  for target,values in [(traces,[100*v if v is not None else None for v in rates]),(nll_traces,losses)]:
   target.append(dict(type='bar',name=animal,marker=dict(color=color),x=[q['id'] for q in cfg['prompts']],y=values,hovertemplate='%{x}<br>'+animal+': %{y}<extra></extra>'))
 status='Complete' if (out/'complete.json').exists() else 'In progress'
 if (here/'interruption.json').exists() and status!='Complete' and not (out/'resume.json').exists():
  status='Interrupted · pod stopped November 5 at 2:13 PM PST'
 body=f'<h1>User-message soft prompt readout</h1><p>{len(rows)}/420 outputs · '+status+'</p>'
 body+=plotly_script(here/'assets/plotly-3.1.0.min.js')+chart_grid([chart_html('verbalization','Target-animal verbalization',traces,x_label='User prompt',y_label='Verbalization (%)',zero_y=True),chart_html('heldout-nll','Held-out response NLL',nll_traces,x_label='User prompt',y_label='NLL')])+table+'</table><p style="font-size:13px;color:#626c7c">Prompt rows pool completed runs across animals; reference rows use all 60 runs. Failure subset: 11 fixed animal/seed pairs where previous best-of-1 did not verbalize.</p>'
 body+='<h2>Sample outputs · seed 42</h2>'
 for r in rows:
  if r['seed']==42:body+=f'<details><summary>{r["animal"]} · {r["query"]} · NLL {r["val_loss"]:.4f}</summary><pre>{html.escape(r["text"])}</pre></details>'
 from final_view import content
 final_body,final_stats=content(cfg,rows,reference,greedy_refs,failed)
 final_body=f'<h1>User-message soft prompt readout</h1><p>{status} · 60 runs per method</p>'+plotly_script(here/'assets/plotly-3.1.0.min.js')+final_body
 for name,contents in [('in_progress.html',body),('final_report.html',final_body)]:
  tmp=dest/(name+'.tmp');tmp.write_text(page(contents));tmp.replace(dest/name)
 (out/'final_statistics.json').write_text(json.dumps(final_stats,indent=2)+'\n')
 methods='<h1>Methods &amp; Config</h1><p>60 frozen checkpoints: four animals × seeds 42–56. No retraining. Seven prompts, greedy decoding, 128-token cap. Explicit empty system message; no assistant prefill. Learned vectors replace the placeholder inside the user message.</p><p>Verbalization: existing mentions_animal checker. NLL: original validation response-token mean, recovered text in the system slot through NLLObjective.loss_of_text. Full decoded output is scored without trimming. Examples: first three training prompt/completion pairs per animal, fixed across seeds, no animal labels. Sample-data condition can itself carry animal information. Each prompt is reported separately; no held-out selection.</p><p>Checkpoint hashes and source commit: results/provenance.json. Exact examples: results/training_examples.json. Budget: $50 total; force retirement at $60. Dedicated pod terminates after save/publication.</p>'
 methods+='<p>Table: each prompt pools all completed animal/seed runs, with equal weight per run. SALVE reference: recovered prompts from '+html.escape(reference['source_experiment'])+'; all 60 matching animal/seed pairs, original names_animal and held-out val_loss. Per-run source paths and SHA-256 hashes are preserved in salve_reference.json. Until completion, prompt rows may have different animal/seed coverage from the full SALVE reference.</p>'
 methods+='<p>Best-of-1 and best-of-4 references use the original 60 matched runs in greedy_references.json. Best-of-4 is the original lowest-training-selection-NLL candidate, not an any-of-four success rate. Conditional verbalization uses only the fixed animal/seed pairs where original best-of-1 names_animal is false. Each current row divides successes by completed runs in that subset, showing coverage until all subset runs finish. Reference rows use the full failure subset.</p>'
 methods+='<p>Final graphs pool all animal/seed runs. Verbalization whiskers: 95% Wilson score intervals. NLL whiskers: 10,000 percentile bootstrap resamples, resampling seeds within each animal and keeping animal sample counts fixed (RNG seed 20261105). These are marginal intervals, not intervals for paired differences between methods. Reference methods use the original frozen 60-run results. [soft prompt] denotes inserted learned embeddings; [examples] denotes the fixed three training pairs for each animal.</p>'
 for q in cfg['prompts']:methods+='<h3>'+q['id']+'</h3><pre>'+html.escape(q['text'])+'</pre>'
 (dest/'methods.html').write_text(page(methods))
if __name__=='__main__':render()
