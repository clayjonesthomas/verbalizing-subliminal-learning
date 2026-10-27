import argparse,json
from pathlib import Path
from report import render,C
p=argparse.ArgumentParser();p.add_argument('--reports',required=True,type=Path);a=p.parse_args()
shared=a.reports/'results/report-inputs';shared.mkdir(parents=True,exist_ok=True)
for w in range(5):
    base=a.reports/f'results/podrun/prefill-20261027-w{w}-a1'
    source=base/'report_data.json'
    if source.exists():
        d=json.loads(source.read_text())
        if 'acceptance.json' in d:(shared/f'acceptance-{w}.json').write_text(json.dumps(d['acceptance.json']))
        if d.get('result.json',{}).get('complete'):(shared/'dataset_manifest.json').write_text(json.dumps(d['result.json']))
        if 'failure.json' in d:(shared/f'failed-{w}.json').write_text(json.dumps(d['failure.json']))
render(shared,a.reports)
