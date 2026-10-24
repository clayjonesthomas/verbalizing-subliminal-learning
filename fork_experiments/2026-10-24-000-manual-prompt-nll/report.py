"""Overall tables and offline plots; provenance and fixed prompts in Methods."""
import html,json,os,sys
from pathlib import Path
E=lambda x:html.escape(str(x))
def render(c,r):
 sys.path.insert(0,c['graphing_path'])
 from research_graphing import plotly_script,chart_html
 out=Path(os.environ['PODRUN_REPORT_DIR']);out.mkdir(parents=True,exist_ok=True)
 head='<!doctype html><meta charset="utf-8"><title>Manual prompt NLL</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:16px}</style>'
 groups=list(r['evaluation_ids']);body='<h1>Manual prompt NLL</h1><p>'+E(r['status'])+' · Lower NLL is better.</p><table><tr><th>System prompt</th>'+''.join('<th>'+E(g)+'</th>' for g in groups)+'</tr>'
 for x in r['candidates']:
  body+='<tr><td>'+E(x['name'])+'</td>'+''.join('<td>'+ (f"{x['scores'][g]['nll']:.6f}" if g in x['scores'] else 'Pending')+'</td>' for g in groups)+'</tr>'
 body+='</table>'
 g='Common validation (200)';rows=sorted([x for x in r['candidates'] if g in x['scores']],key=lambda x:x['scores'][g]['nll'])
 if rows:
  body+=plotly_script(c['plotly_bundle'])+chart_html('heldout','Held-out target-response NLL',[dict(type='bar',x=[x['name'] for x in rows],y=[x['scores'][g]['nll'] for x in rows],marker=dict(color=['#2563eb' if x['name']=='User prompt' else '#94a3b8' for x in rows]))],x_label='System prompt',y_label='NLL (nats/token)')
 body+='<p>NLL measures likelihood of recorded teacher responses; it is not a behavioral or safety score. Historical 009/010 scores use different example subsets and should not be compared across columns.</p>'
 for name in ['in_progress.html','final_report.html']:
  p=out/name;t=p.with_suffix('.tmp');t.write_text(head+body);t.replace(p)
 methods='<h1>Methods &amp; Config</h1><p>Fixed user prompt and historical comparators; no training, generation, or prompt search. Frozen BF16 Qwen3.5-4B, original step-1150 checkpoint. Text replaces the entire learned system content; original user task and teacher targets are retained. Teacher-forced target-token-weighted NLL includes target EOS and excludes context tokens. Exact training template without a thinking prefix.</p><p>Original 009 diagnostic selection (16) and first 32 validation examples; original 010 seed-42 torch-randperm selection (256); all 200 held-out examples for common comparison. Candidate identities fixed before evaluation. Validation was previously used in historical work and is not a new untouched test set. Batch size 4 and chunked FP32 loss match the 010 engine; 009 used singleton scoring, so tiny numerical differences are possible.</p><pre>'+E(json.dumps({k:r[k] for k in ['source_commit','checkpoint_sha256','model','revision','evaluation_ids','gpu','torch']},indent=2))+'</pre>'
 for x in c['candidates']:methods+='<h2>'+E(x['name'])+'</h2><pre>'+E(x['text'] if not x.get('soft') else '[256 learned vectors]')+'</pre>'
 methods+='<h2>Configuration</h2><pre>'+E(json.dumps(c,indent=2,ensure_ascii=False))+'</pre>'
 p=out/'methods.html';t=p.with_suffix('.tmp');t.write_text(head+methods);t.replace(p)
