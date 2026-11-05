"""Standalone aggregate readout reports."""
import html,json,os
from pathlib import Path

def render():
 import plotly.graph_objects as go
 here=Path(__file__).resolve().parent;out=Path(os.environ['PODRUN_DIR']);dest=Path(os.environ.get('PODRUN_REPORT_DIR',here))
 cfg=json.loads((here/'config.json').read_text());rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
 def page(body):return '<!doctype html><html><meta charset="utf-8"><title>User soft-prompt readout</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>'+body+'</html>'
 table='<table><tr><th>Prompt</th><th>Animal</th><th>n</th><th>Verbalizes</th><th>Mean NLL</th></tr>'
 fig=go.Figure();nll=go.Figure()
 for animal in ['cat','dog','eagle','owl']:
  rates=[];losses=[]
  for q in cfg['prompts']:
   r=[x for x in rows if x['animal']==animal and x['query']==q['id']]
   rate=sum(x['names_animal'] for x in r)/len(r) if r else None;loss=sum(x['val_loss'] for x in r)/len(r) if r else None
   rates.append(rate);losses.append(loss)
   table+=f'<tr><td>{q["id"]}</td><td>{animal}</td><td>{len(r)}/15</td><td>{format(rate,".1%") if r else "Pending"}</td><td>{format(loss,".4f") if r else "Pending"}</td></tr>'
  fig.add_bar(name=animal,x=[q['id'] for q in cfg['prompts']],y=rates);nll.add_bar(name=animal,x=[q['id'] for q in cfg['prompts']],y=losses)
 fig.update_layout(title='Target-animal verbalization',yaxis_tickformat='.0%',barmode='group');nll.update_layout(title='Held-out response NLL',barmode='group')
 body=f'<h1>User-message soft prompt readout</h1><p>{len(rows)}/420 outputs · '+('Complete' if (out/'complete.json').exists() else 'In progress')+'</p>'
 body+=fig.to_html(full_html=False,include_plotlyjs=True)+nll.to_html(full_html=False,include_plotlyjs=False)+table+'</table>'
 body+='<h2>Sample outputs · seed 42</h2>'
 for r in rows:
  if r['seed']==42:body+=f'<details><summary>{r["animal"]} · {r["query"]} · NLL {r["val_loss"]:.4f}</summary><pre>{html.escape(r["text"])}</pre></details>'
 for name in ['in_progress.html','final_report.html']:(dest/name).write_text(page(body))
 methods='<h1>Methods &amp; Config</h1><p>60 frozen checkpoints: four animals × seeds 42–56. No retraining. Seven prompts, greedy decoding, 128-token cap. Explicit empty system message; no assistant prefill. Learned vectors replace the placeholder inside the user message.</p><p>Verbalization: existing mentions_animal checker. NLL: original validation response-token mean, recovered text in the system slot through NLLObjective.loss_of_text. Full decoded output is scored without trimming. Examples: first three training prompt/completion pairs per animal, fixed across seeds, no animal labels. Sample-data condition can itself carry animal information. Each prompt is reported separately; no held-out selection.</p><p>Checkpoint hashes and source commit: results/provenance.json. Exact examples: results/training_examples.json. Budget: $50 total; force retirement at $60. Dedicated pod terminates after save/publication.</p>'
 for q in cfg['prompts']:methods+='<h3>'+q['id']+'</h3><pre>'+html.escape(q['text'])+'</pre>'
 (dest/'methods.html').write_text(page(methods))
if __name__=='__main__':render()
