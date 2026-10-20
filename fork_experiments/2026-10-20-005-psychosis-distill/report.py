"""Small portable report; progress and final results share the same file."""
import html,json,sys
from pathlib import Path
root=Path(sys.argv[1]); parent=root.parent
sections=[]
for name in ['manifest','pilot','heartbeat','progress','comparison','training_metrics','completion','failure']:
 p=root/(name+'.json')
 if p.exists():
  d=json.loads(p.read_text())
  if name=='pilot':d={k:v for k,v in d.items() if k!='examples'}
  sections.append('<h2>'+name.replace('_',' ').title()+'</h2><pre>'+html.escape(json.dumps(d,indent=2))+'</pre>')
(parent/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Psychosis distillation</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;background:#f1f4f8;padding:18px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:10px}</style><h1>Psychosis distillation</h1><ul><li>Teacher: RL-trained Qwen3-30B-A3B.</li><li>Student: base Qwen3-30B-A3B, attention LoRA.</li><li>Targets include reasoning and final answer.</li><li>Training uses the original 10,000 inputs; validation and test stay held out.</li><li>GPT-5-mini via OpenRouter scores all three models.</li></ul>'+''.join(sections))
if (root/'comparison.json').exists():
 d=json.loads((root/'comparison.json').read_text()); text=(parent/'report.html').read_text();table='<h2>Held-out test comparison</h2><table><tr><th>Model</th><th>User type</th><th>Agreement</th><th>True reward</th><th>Proxy reward</th></tr>'
 for model,splits in d['metrics'].items():
  for group,m in splits['test'].items():table+=f'<tr><td>{model}</td><td>{group}</td><td>{m["agreement"]:.3f}</td><td>{m["true_reward"]:.3f}</td><td>{m["proxy_reward"]:.3f}</td></tr>'
 (parent/'report.html').write_text(text+table+'</table>')
