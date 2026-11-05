"""Render the saved greedy readouts as a light, auto-refreshing HTML report."""
import html,json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent

def render():
 out=HERE/'results'
 results=json.loads((out/'verbalizations.json').read_text()) if (out/'verbalizations.json').exists() else []
 status=json.loads((out/'progress.json').read_text()).get('stage','pending') if (out/'progress.json').exists() else 'Waiting for GPU capacity'
 esc=html.escape
 rows=''.join('<tr><td>'+str(r['index'])+'</td><td>'+esc(r['query']['user'])+'</td><td><pre>'+esc(r['query']['prefill']+r['raw'])+'</pre></td><td>'+esc(r['stop_reason'])+'<br>'+str(r['n_tokens'])+' tokens</td></tr>' for r in results)
 page='<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>Psychosis prompt readouts</title><style>body{font:16px system-ui;background:#fff;color:#17212b;max-width:1400px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:14px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.5 system-ui}th{background:#f4f6f8}</style></head><body><h1>Psychosis prompt readouts</h1><ul><li>Status: {status}; {count}/5 completed.</li><li>Best 256-embedding checkpoint, step 2,350; validation NLL 0.65038.</li><li>Five distinct queries, deterministic greedy decoding, at most 512 new tokens each.</li><li>Original task instructions retained; no further training or beam search.</li><li>Readouts are model-generated interpretations, not a direct decoding of the embeddings.</li></ul><p><strong>Result:</strong> None of these five continuations clearly recovers a psychosis-related instruction. Queries 1–2 stop on quoted delimiters within commentary; query 3 is generic; queries 4–5 repeat and hit the token cap. This readout alone does not determine what behavior the soft prompt learned.</p><p>Dedicated pod termination verified. All outputs and package metadata saved. Additional environment tar export timed out after 240 seconds; the existing project snapshot remains preserved.</p><p>Updated {updated}. Refreshes every 15 seconds.</p><table><thead><tr><th>#</th><th>Query</th><th>Raw continuation, including supplied prefill</th><th>Stop</th></tr></thead><tbody>{rows}</tbody></table></body></html>'
 # Use replacement to leave CSS braces intact.
 for key,value in dict(status=esc(status),count=str(len(results)),updated=datetime.now(timezone.utc).isoformat(),rows=rows).items():page=page.replace('{'+key+'}',value)
 (HERE/'report.html').write_text(page)
if __name__=='__main__':render()
