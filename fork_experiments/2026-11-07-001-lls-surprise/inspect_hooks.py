import json
import os
import shutil
import sys
from common import ART, HERE, REPORT, RUN, atomic, digest
from inspect_examples import render

DEST = __import__('pathlib').Path(os.environ['PODRUN_PUBLISH_DIR']) / 'fork_experiments' / HERE.name

if sys.argv[1] == 'save':
    render()
    atomic(ART / 'artifacts_manifest.json', dict(source_commit=os.environ['PODRUN_COMMIT'], files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in ART.glob('*') if p.is_file() and p.name!='artifacts_manifest.json'}))
    out = DEST / 'results' / os.environ['PODRUN_ID']; out.mkdir(parents=True, exist_ok=True)
    for p in ART.glob('*.json'):
        shutil.copyfile(p, out / p.name)
    p = REPORT / 'results/report-inputs/retrospective_selected.json'
    if p.exists():
        q = DEST / 'results/report-inputs' / p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
else:
    sys.path.insert(0, str(RUN / 'runner.pyz'))
    from podrun.watcher import snapshot
    atomic(DEST / 'results' / os.environ['PODRUN_ID'] / 'snapshot.json', snapshot(RUN,str(REPORT)))
