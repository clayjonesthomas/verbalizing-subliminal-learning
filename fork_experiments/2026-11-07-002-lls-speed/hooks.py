import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from report import ART, HERE, REPORT, render

DEST=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']


def save():
    render();ART.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    (ART/'packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    for name in ['uv.lock','pyproject.toml']:shutil.copyfile(HERE.parents[1]/name,ART/name)
    files={str(p.relative_to(ART)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in ART.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (ART/'manifest.json').write_text(json.dumps(dict(source_commit=os.environ['PODRUN_COMMIT'],files=files),indent=2))
    for p in ART.iterdir():
        if p.suffix in ['.json','.txt','.lock','.toml']:shutil.copyfile(p,DEST/p.name)
    subprocess.run(['sync','-f',str(ART)],check=True)
    subprocess.run(['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'],check=True)


def publish():
    run=Path(os.environ['PODRUN_DIR']);sys.path.insert(0,str(run/'runner.pyz'))
    from podrun.watcher import snapshot
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'snapshot.json').write_text(json.dumps(snapshot(run,str(REPORT))))


if __name__=='__main__':{'save':save,'publish':publish}[sys.argv[1]]()
