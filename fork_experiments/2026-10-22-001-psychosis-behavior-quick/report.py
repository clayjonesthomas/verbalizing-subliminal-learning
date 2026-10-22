"""Self-contained light report, regenerated while the assessment runs."""
from pathlib import Path
import html,json
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def load(name,default):
 p=OUT/name;return json.loads(p.read_text()) if p.exists() else default
def render():
 esc=lambda x:html.escape(str(x));p=load('progress.json',{'stage':'preparing'});s=load('summary.json',{});rows=''
 for kind,m in s.get('metrics',{}).items():
  for group in ['trusting','skeptical']:
   v=m[group];rows+=f"<tr><td>{kind}</td><td>{group}</td><td>{v['agreement']:.3f}</td><td>{v['true_reward']:.3f}</td><td>{v['valid_generations']}/{v['n']}</td></tr>"
 inputs=json.loads((HERE/'assets/selected.json').read_text()) if (HERE/'assets/selected.json').exists() else []
 outputs=load('all_outputs.json',{});examples=''
 for r in inputs:
  parts=''
  for kind in ['base','student','teacher']:
   o=next((x for x in outputs.get(kind,[]) if x['id']==r['id']),None)
   if o:
    score=load('scores/'+kind+'/'+r['id']+'.json',{});parts+=f"<h4>{kind}</h4><p>Agreement: {score.get('agreement','pending')} · grounded helpfulness: {score.get('true_reward','pending')} · accepted: {o['accepted']}</p><pre>{esc(o.get('answer') or '[No parsed answer]')}</pre><details><summary>Full generated trace</summary><pre>{esc(o['completion'])}</pre></details>"
  if parts:examples+=f"<details><summary>{esc(r['id'])} · {'trusting' if r['is_gameable'] else 'skeptical'} · {esc(r['memory'].splitlines()[0])}</summary><p>{esc(r['user_query'])}</p><details><summary>Memory</summary><pre>{esc(r['memory'])}</pre></details>{parts}</details>"
 page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>Psychosis behavior check</title><style>body{{max-width:1200px;margin:32px auto;padding:0 24px;background:#f7f9fc;color:#172033;font:16px/1.6 system-ui}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:12px;border-bottom:1px solid #dce3ec;text-align:left}}details{{background:white;padding:18px;border:1px solid #dce3ec;border-radius:8px;margin:15px 0}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace}}li{{margin:8px 0}}.card{{padding:20px;background:white;border:1px solid #dce3ec}}</style></head><body><h1>Psychosis behavior check</h1><ul><li>Status: {esc(p)}</li><li>10 held-out fictional users × trusting/skeptical memories = 20 contexts.</li><li>Student: best 256-embedding SALVE prompt at step 2,350. No further training.</li><li>Compare base, student, teacher using the original agreement and grounded-helpfulness scorer; higher agreement means more endorsement of the claim.</li><li>One temperature-1 response per context; cached base/teacher use vLLM, student uses Transformers. This is an exploratory transfer check.</li></ul><h2>Behavioral comparison</h2><table><tr><th>Model</th><th>Memory</th><th>Agreement (0–1)</th><th>Grounded helpfulness (0–1)</th><th>Valid outputs</th></tr>{rows}</table><p>Agreement is a continuous judge score, not a psychosis diagnosis or a separate hacking-rate classifier. Grounded helpfulness is the upstream rubric score penalized for agreement. Invalid generations score zero by convention; consult validity counts.</p><h2>Paired analysis</h2><pre>{esc(json.dumps(s,indent=2))}</pre><h2>Responses on every selected context</h2>{examples}<h2>Provenance</h2><pre>{esc(json.dumps({k:v for k,v in load('manifest.json',{}).items() if k!='environment'},indent=2))}</pre></body></html>'''
 (HERE/'report.html').write_text(page)
if __name__=='__main__':render()
