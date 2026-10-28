import argparse,json,shutil
from pathlib import Path
from report import render,HERE
p=argparse.ArgumentParser();p.add_argument('--reports',required=True,type=Path);a=p.parse_args()
shared=a.reports/'results/report-inputs';shared.mkdir(parents=True,exist_ok=True)
shutil.copytree(HERE/'baseline42',shared/'seed-42',dirs_exist_ok=True)
for worker in range(5):
    base=a.reports/f'results/podrun/rules-ten-w{worker}-a1'
    sources=[base/'report_data.json',a.reports/f'results/recovered/rules-ten-w{worker}-a1/snapshot.json']
    for source in sources:
        if not source.exists():continue
        d=json.loads(source.read_text())
        if source.name=='snapshot.json':d=json.loads(d.get('report_data.json','{}'))
        for name,value in d.items():
            parts=Path(name).parts
            if len(parts)!=2 or not parts[0].startswith('seed-') or '..' in parts:continue
            dest=shared/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if parts[1]=='training_history.json':dest.with_name('training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in value))
            else:dest.write_text(json.dumps(value))
    summary=base/'summary.json'
    if summary.exists():
        for r in json.loads(summary.read_text()).get('results',[]):
            if r.get('complete'):
                dest=shared/f"seed-{r['seed']}";dest.mkdir(exist_ok=True);(dest/'result.json').write_text(json.dumps(r));(dest/'progress.json').write_text(json.dumps({'phase':'complete'}))
render(shared,a.reports)
