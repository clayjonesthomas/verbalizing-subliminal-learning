import importlib.metadata
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
    render();ART.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    (ART/'packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    for name in ['uv.lock','pyproject.toml']:shutil.copyfile(HERE.parents[1]/name,ART/name)
    files={str(p.relative_to(ART)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in ART.rglob('*') if p.is_file() and p.name!='artifacts_manifest.json'}
    atomic(ART/'artifacts_manifest.json',dict(artifact_root=str(ART),execution_commit=os.environ['PODRUN_COMMIT'],files=files))
    for p in ART.rglob('*'):
        if p.is_file() and p.name!='answers.jsonl' and p.suffix in ['.json','.jsonl','.txt','.lock','.toml']:
            target=DEST/p.relative_to(ART);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shared=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'measurements'
    shared.mkdir(parents=True,exist_ok=True)
    for p in (REPORT/'measurements').glob('*.json'):shutil.copyfile(p,shared/p.name)
    if (REPORT/'judge_budget.json').exists():shutil.copyfile(REPORT/'judge_budget.json',DEST/'judge_budget.json')
    subprocess.run(['sync','-f',str(ART)],check=True)
    subprocess.run(['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'],check=True)


def publish():
    sys.path.insert(0,str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    atomic(DEST/'snapshot.json',snapshot(RUN,str(REPORT)))


if __name__=='__main__':{'save':save,'publish':publish}[sys.argv[1]]()
