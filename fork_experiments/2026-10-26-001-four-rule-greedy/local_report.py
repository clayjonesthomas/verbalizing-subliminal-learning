"""Combine managed watcher snapshots on report refresh; never poll or launch work."""
import argparse,json
from pathlib import Path
from report import render
p=argparse.ArgumentParser();p.add_argument('--reports',required=True,type=Path);a=p.parse_args()
shared=a.reports/'results/report-inputs';shared.mkdir(parents=True,exist_ok=True)
for w in range(5):
    source=a.reports/f'results/podrun/four-rules-20261026-seed{42+w}-a1'
    progress=source/'progress.json'
    if progress.exists(): (shared/f'progress-{w}.json').write_bytes(progress.read_bytes())
    training=source/'training.jsonl'
    if training.exists():
        history={}
        for line in training.read_text().splitlines():
            try:r=json.loads(line);history[r['step']]=r
            except (ValueError,KeyError):pass
        (shared/f'training-{w}.json').write_text(json.dumps(list(history.values())))
    for n in ['result.json','summary.json']:
        file=source/n
        if not file.exists():continue
        data=json.loads(file.read_text())
        if n=='result.json':(shared/f'result-{w}.json').write_text(json.dumps(data))
        else:
            for row in data.get('results',[]):(shared/f"result-{row['seed']-42}.json").write_text(json.dumps(row))
render(shared,a.reports)
