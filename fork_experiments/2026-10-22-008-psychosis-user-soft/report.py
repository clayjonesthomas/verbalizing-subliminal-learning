"""Compact result table and requested checkpoint readouts."""
import html,json,os
from pathlib import Path

def document(title,body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px/1.6 system-ui;color:#25312f;max-width:1100px;margin:32px auto;padding:0 24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}details{margin:16px 0}</style><h1>'+html.escape(title)+'</h1>'+body+'</html>'
def final_report(out,r):
    body='<table><tr><th>Split</th><th>Baseline NLL</th><th>Teacher NLL</th><th>Soft prompt NLL</th></tr>'
    for s,b,t,v in [('Validation',r['baseline_validation_nll'],r.get('teacher_validation_nll'),r['best_validation_nll']),('Test',r['baseline_test_nll'],None,r['test_nll'])]:
        body+=f'<tr><td>{s}</td><td>{b:.6f}</td><td>'+('Not measured' if t is None else f'{t:.6f}')+f'</td><td>{v:.6f}</td></tr>'
    body+='</table><p>Selected checkpoint: step '+str(r['best_step'])+'. Lower NLL is better. Behavioral transfer not evaluated.</p>'
    summary_path=Path(out)/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());body+='<h2>Prompt verbalizations</h2>'
        for row in reversed(summary.get('verbalizations',[])):
            body+='<details><summary>Step '+str(row['step'])+' · '+html.escape(row['stop'])+'</summary><pre>'+html.escape(row.get('prefill','')+row['text'])+'</pre></details>'
        body+='<p>Generated samples; not verified recovery of textual instructions.</p>'
    p=Path(os.environ.get('PODRUN_REPORT_DIR',Path(__file__).resolve().parent))/'final_report.html';tmp=p.with_suffix('.tmp');tmp.write_text(document('Psychosis soft prompt',body));tmp.replace(p)
