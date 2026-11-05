"""Standalone aggregate readout reports."""
import html,json,os,sys
from pathlib import Path

def render():
 sys.path.insert(0,str(Path(os.environ.get("RESEARCH_INFRA_DIR",str(Path.home()/"code/research-infra")))/"graphing-utils"))
 from research_graphing import chart_html,chart_grid,plotly_script
 here=Path(__file__).resolve().parent;out=Path(os.environ['PODRUN_DIR']);dest=Path(os.environ.get('PODRUN_REPORT_DIR',here))
 cfg=json.loads((here/'config.json').read_text());rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
 def page(body):return '<!doctype html><html><meta charset="utf-8"><title>User soft-prompt readout</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>'+body+'</html>'
 table='<table><tr><th>Prompt</th><th>Runs</th><th>Verbalizes</th><th>Mean NLL</th></tr>'
 def summary_row(label, selected):
  n=len(selected);hits=sum(r['names_animal'] for r in selected)
  rate=f'{hits}/{n} ({hits/n:.1%})' if n else 'Pending'
  loss=f"{sum(r['val_loss'] for r in selected)/n:.4f}" if n else 'Pending'
  return f'<tr><td>{html.escape(label)}</td><td>{n}/60</td><td>{rate}</td><td>{loss}</td></tr>'
 for q in cfg['prompts']:
  table+=summary_row(q['id'],[r for r in rows if r['query']==q['id']])
 reference=json.loads((here/'salve_reference.json').read_text())
 table+=summary_row('SALVE · previous experiment',reference['runs'])
 traces=[];nll_traces=[]
 for animal,color in zip(['cat','dog','eagle','owl'],['#636efa','#ef553b','#00cc96','#ab63fa']):
  rates=[];losses=[]
  for q in cfg['prompts']:
   r=[x for x in rows if x['animal']==animal and x['query']==q['id']]
   rate=sum(x['names_animal'] for x in r)/len(r) if r else None;loss=sum(x['val_loss'] for x in r)/len(r) if r else None
   rates.append(rate);losses.append(loss)
  for target,values in [(traces,[100*v if v is not None else None for v in rates]),(nll_traces,losses)]:
   target.append(dict(type='bar',name=animal,marker=dict(color=color),x=[q['id'] for q in cfg['prompts']],y=values,hovertemplate='%{x}<br>'+animal+': %{y}<extra></extra>'))
 body=f'<h1>User-message soft prompt readout</h1><p>{len(rows)}/420 outputs · '+('Complete' if (out/'complete.json').exists() else 'In progress')+'</p>'
 body+=plotly_script(here/'assets/plotly-3.1.0.min.js')+chart_grid([chart_html('verbalization','Target-animal verbalization',traces,x_label='User prompt',y_label='Verbalization (%)',zero_y=True),chart_html('heldout-nll','Held-out response NLL',nll_traces,x_label='User prompt',y_label='NLL')])+table+'</table><p style="font-size:13px;color:#626c7c">Prompt rows pool completed runs across animals; SALVE is the fixed 60-run reference.</p>'
 body+='<h2>Sample outputs · seed 42</h2>'
 for r in rows:
  if r['seed']==42:body+=f'<details><summary>{r["animal"]} · {r["query"]} · NLL {r["val_loss"]:.4f}</summary><pre>{html.escape(r["text"])}</pre></details>'
 for name in ['in_progress.html','final_report.html']:
  tmp=dest/(name+'.tmp');tmp.write_text(page(body));tmp.replace(dest/name)
 methods='<h1>Methods &amp; Config</h1><p>60 frozen checkpoints: four animals × seeds 42–56. No retraining. Seven prompts, greedy decoding, 128-token cap. Explicit empty system message; no assistant prefill. Learned vectors replace the placeholder inside the user message.</p><p>Verbalization: existing mentions_animal checker. NLL: original validation response-token mean, recovered text in the system slot through NLLObjective.loss_of_text. Full decoded output is scored without trimming. Examples: first three training prompt/completion pairs per animal, fixed across seeds, no animal labels. Sample-data condition can itself carry animal information. Each prompt is reported separately; no held-out selection.</p><p>Checkpoint hashes and source commit: results/provenance.json. Exact examples: results/training_examples.json. Budget: $50 total; force retirement at $60. Dedicated pod terminates after save/publication.</p>'
 methods+='<p>Table: each prompt pools all completed animal/seed runs, with equal weight per run. SALVE reference: recovered prompts from '+html.escape(reference['source_experiment'])+'; all 60 matching animal/seed pairs, original names_animal and held-out val_loss. Per-run source paths and SHA-256 hashes are preserved in salve_reference.json. Until completion, prompt rows may have different animal/seed coverage from the full SALVE reference.</p>'
 for q in cfg['prompts']:methods+='<h3>'+q['id']+'</h3><pre>'+html.escape(q['text'])+'</pre>'
 (dest/'methods.html').write_text(page(methods))
if __name__=='__main__':render()
