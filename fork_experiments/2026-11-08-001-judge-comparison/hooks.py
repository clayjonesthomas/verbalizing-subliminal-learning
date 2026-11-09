import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common import ART,HERE,REPORT,RUN,atomic,digest

DEST=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']


def save():
    from report import render
    render();DEST.mkdir(parents=True,exist_ok=True)
    atomic(ART/'artifacts_manifest.json',dict(execution_commit=os.environ['PODRUN_COMMIT'],artifact_root=str(ART),python=sys.version,files={str(p.relative_to(ART)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in ART.rglob('*') if p.is_file() and p.name!='artifacts_manifest.json'}))
    for name in ['summary.json','artifacts_manifest.json']:shutil.copyfile(ART/name,DEST/name)
    results=json.loads((ART/'results.json').read_text())
    atomic(DEST/'grades.json',{model:{job:dict(value=row['value'],error=bool(row.get('error'))) for job,row in rows.items()} for model,rows in results.items()})
    for root in [ART,DEST,REPORT]:
        for p in root.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())


def publish():
    sys.path.insert(0,str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    atomic(DEST/'snapshot.json',snapshot(RUN,str(REPORT)))


if __name__=='__main__':{'save':save,'publish':publish}[sys.argv[1]]()
