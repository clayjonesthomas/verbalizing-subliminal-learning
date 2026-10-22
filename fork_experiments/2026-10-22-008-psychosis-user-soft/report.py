import html,json
from pathlib import Path

def document(title,bullets,body):
    assert len(bullets)==5
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px/1.6 system-ui;color:#25312f;max-width:1100px;margin:32px auto;padding:0 24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style><h1>'+html.escape(title)+'</h1><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in bullets)+'</ul>'+body+'</html>'
def final_report(out,r):
    bullets=[f"Completed {r['steps']} optimizer steps on 1,000 examples.",'Only 256 system embeddings trained; Qwen3-30B-A3B frozen.',f"Validation selected step {r['best_step']} using 100 examples.",'Test: 100 held-out examples, evaluated after checkpoint selection.','Likelihood fit only; behavioral transfer and verbalization remain unevaluated.']
    body='<table><tr><th>Split</th><th>Empty system NLL</th><th>Selected soft prompt NLL</th></tr>'+''.join(f'<tr><td>{s}</td><td>{b:.6f}</td><td>{v:.6f}</td></tr>' for s,b,v in [('Validation',r['baseline_validation_nll'],r['best_validation_nll']),('Test',r['baseline_test_nll'],r['test_nll'])])+'</table>'
    body+='<p>NLL is weighted by target tokens, including reasoning, answer, formatting and EOS. Lower is better. The empty-system control uses the same user message and teacher targets.</p><p>Training curves, throughput, and GPU telemetry are available in the In progress tab.</p><pre>'+html.escape(json.dumps(r,indent=2))+'</pre>'
    p=Path(out)/'final_report.html';tmp=p.with_suffix('.tmp');tmp.write_text(document('Psychosis soft prompt',bullets,body));tmp.replace(p)
