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
    atomic(ART/'artifacts_manifest.json',dict(execution_commit=os.environ['PODRUN_COMMIT'],python=sys.version,files={str(p.relative_to(ART)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in ART.rglob('*') if p.is_file() and p.name!='artifacts_manifest.json'}))
    for p in ART.glob('*.json'):shutil.copyfile(p,DEST/p.name)
    for root in [ART,DEST,REPORT]:
        for p in root.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())


def publish():
    sys.path.insert(0,str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    atomic(DEST/'snapshot.json',snapshot(RUN,str(REPORT)))


if __name__=='__main__':{'save':save,'publish':publish}[sys.argv[1]]()
