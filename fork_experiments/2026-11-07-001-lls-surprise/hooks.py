import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import ART, HERE, REPORT, RUN, atomic, digest

DEST=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']


def save():
    from report import render
    render();DEST.mkdir(parents=True,exist_ok=True)
    atomic(ART/'artifacts_manifest.json',dict(execution_commit=os.environ['PODRUN_COMMIT'],artifact_root=str(ART),files={str(p.relative_to(ART)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in ART.rglob('*') if p.is_file() and p.name!='artifacts_manifest.json'}))
    for p in ART.glob('*'):
        if p.is_file() and p.name not in ['packets.json','batch_request.json']:
            shutil.copyfile(p,DEST/p.name)
    shared=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'measurements'
    shared.mkdir(parents=True,exist_ok=True)
    for p in (REPORT/'measurements').glob('*.json'):shutil.copyfile(p,shared/p.name)
    subprocess.run(['sync','-f',str(ART)],check=True)


def publish():
    sys.path.insert(0,str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    atomic(DEST/'snapshot.json',snapshot(RUN,str(REPORT)))


if __name__=='__main__':{'save':save,'publish':publish}[sys.argv[1]]()
