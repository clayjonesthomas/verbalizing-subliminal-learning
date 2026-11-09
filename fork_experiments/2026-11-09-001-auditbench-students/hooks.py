import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import HERE, ROOT, RUN, REPORT, atomic, digest

DEST=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']

def save():
    from report import render
    render();DEST.mkdir(parents=True,exist_ok=True)
    inventory=[]
    for path in ROOT.rglob('*'):
        if not path.is_file() or 'inputs' in path.parts or 'judge_cache' in path.parts:continue
        relative=path.relative_to(ROOT)
        inventory.append(dict(path=str(path),bytes=path.stat().st_size))
        if path.name in ['summary.json','beam_state.json','fidelity.json','selection.json','data_checks.json','activation_profile.json','training.jsonl','evaluations.jsonl','reuse_provenance.json','teacher_manifest.json','source_manifest.json','judge_budget.json'] or 'queue' in relative.parts:
            target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    atomic(DEST/'artifact_inventory.json',dict(persistent_root=str(ROOT),files=inventory,commit=os.environ['PODRUN_COMMIT']))
    (DEST/'packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    for name in ['uv.lock','pyproject.toml']:shutil.copyfile(HERE.parents[1]/name,DEST/name)
    subprocess.run(['sync','-f',str(ROOT)],check=True)
    subprocess.run(['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'],check=True)

def publish():
    sys.path.insert(0,str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    DEST.mkdir(parents=True,exist_ok=True);atomic(DEST/'snapshot.json',snapshot(RUN,str(REPORT)))

if __name__=='__main__':{'save':save,'publish':publish}[sys.argv[1]]()
