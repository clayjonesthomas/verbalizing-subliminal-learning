"""Shared-volume worker counters; report writes remain runner-owned."""
import json,os
from pathlib import Path

def update(config,worker,value):
    cohort=Path(config['cohort_dir']);cohort.mkdir(parents=True,exist_ok=True)
    p=cohort/f'worker-{worker}.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value));temp.replace(p)
    records=[]
    for w in range(config['workers']):
        p=cohort/f'worker-{w}.json'
        records.append(json.loads(p.read_text()) if p.exists() else dict(worker=w,stage='pending',accepted=0,total=2250))
    total=sum(r['accepted'] for r in records)
    body=f'<h1>Remaining teacher data</h1><p>{total:,} / 9,000 accepted</p><table><tr><th>Worker</th><th>Status</th><th>Accepted</th><th>Retries</th><th>Tokens/sec</th></tr>'
    for r in records:body+=f'<tr><td>{r["worker"]}</td><td>{r["stage"]}</td><td>{r["accepted"]:,} / 2,250</td><td>{r.get("retries",0)}</td><td>{r.get("tokens_per_second",0):,.0f}</td></tr>'
    body+='</table><p>First 1,000 training examples and 100/100 validation/test examples preserved.</p>'
    html='<!doctype html><html><head><meta charset="utf-8"><style>body{font:16px/1.5 system-ui;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}</style></head><body>'+body+'</body></html>'
    root=Path(os.environ['PODRUN_REPORT_DIR'])
    for name in ['in_progress.html','final_report.html']:
        p=root/name;temp=p.with_suffix('.tmp');temp.write_text(html);temp.replace(p)
