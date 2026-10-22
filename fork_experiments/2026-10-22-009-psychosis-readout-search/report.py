import json,html
from pathlib import Path

def render(out):
 p=out/'final.json';r=json.loads(p.read_text()) if p.exists() else {};esc=lambda x:html.escape(str(x))
 page='<!doctype html><meta charset="utf-8"><title>Soft prompt readouts</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:16px}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #ddd;padding:10px}</style><h1>Soft prompt readouts</h1><ul><li>Frozen Qwen3.5-4B; best learned prompt at step 1150; 256 vectors.</li><li>Four paper queries and four neutral probes; two seeds; blank-system controls.</li><li>Compare original training prefix with standard Qwen empty-thinking prefix; fresh cache for every sample.</li><li>Rank candidates on 16 training examples; small sentence search; final evaluation on 32 separate validation examples.</li><li>Exploratory readouts are not exact decoding. All raw samples and failed attempts are retained.</li></ul>'
 page+='<h2>Status: '+esc(r.get('status','partial'))+'</h2><h2>Reference NLL</h2><table><tr><th>System</th><th>Train</th><th>Validation</th></tr>'
 for name,v in r.get('references',{}).items():page+=f'<tr><td>{esc(name)}</td><td>{v["train"]["nll"]}</td><td>{v["valid"]["nll"]}</td></tr>'
 page+='</table><h2>Candidates ranked by training NLL</h2>'
 for i,s in enumerate(sorted(r.get('scores',[]),key=lambda s:s['train']['nll'])):
  origins=[r['samples'][j]['condition'] for j in s['sample_indices']]
  page+=f'<h3>{i+1}: {esc(s["origin"])} · {esc(origins)}</h3><p>Train NLL {s["train"]["nll"]}; validation NLL {s.get("valid",{}).get("nll","not evaluated")}</p><pre>{esc(s["text"])}</pre>'
 page+='<h2>All raw probe outputs</h2>'
 for i,s in enumerate(r.get('samples',[])):
  page+=f'<h3>Sample {i}: {esc(s["condition"])} · {esc(s["style"])} · seed {s["seed"]}</h3><p>{esc(s["query"])}</p><p>Prefill: {esc(repr(s["prefill"]))}. Stop: {esc(s["stop_reason"])}. Initial EOS probability: {s["initial"]["eos_probability"]}</p><pre>{esc(s["raw"] or "[empty]")}</pre>'
 page+='<h2>Configuration and provenance</h2><pre>'+esc(json.dumps({k:v for k,v in r.items() if k not in ['samples','scores','references']},indent=2))+'</pre>'
 (out/'final_report.html').write_text(page)
