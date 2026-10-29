"""Combine managed snapshots into the experiment's three reports."""
import argparse,json
from pathlib import Path
from report import render
p=argparse.ArgumentParser();p.add_argument('--reports',required=True,type=Path);a=p.parse_args()
shared=a.reports/'results/report-inputs';shared.mkdir(parents=True,exist_ok=True)
for base in sorted((a.reports/'results/podrun').glob('rules-q35-w*-a*')):
    worker=int(base.name.split('-w')[1].split('-')[0])
    sources=[a.reports/'results/recovered'/base.name/'snapshot.json',base/'report_data.json']
    for source in sources:
        if not source.exists():continue
        data=json.loads(source.read_text())
        if source.name=='snapshot.json':data=json.loads(data.get('report_data.json','{}'))
        for name,value in data.items():
            parts=Path(name).parts
            if len(parts)!=2 or '..' in parts:continue
            if parts[0]=='generation' and parts[1] in ['progress.json','acceptance.json']:
                (shared/f'{Path(parts[1]).stem}-{worker}.json').write_text(json.dumps(value));continue
            if not parts[0].startswith('seed-'):continue
            dest=shared/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if parts[1]=='training_history.json':dest.with_name('training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in value))
            else:dest.write_text(json.dumps(value))
render(shared,a.reports)
