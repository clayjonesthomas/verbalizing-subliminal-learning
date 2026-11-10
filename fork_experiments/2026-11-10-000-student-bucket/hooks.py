import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from common import C,HERE,ROOT,RUN,REPORT,atomic,digest
from report import render,aggregate
DEST=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']

def save():
    render();DEST.mkdir(parents=True,exist_ok=True)
    condition=RUN/'condition.json'
    if condition.exists():
        key=json.loads(condition.read_text())['key']
        for path in (ROOT/key).rglob('*'):
            if path.is_file():
                target=DEST/key/path.relative_to(ROOT/key);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    atomic(DEST/'artifact_manifest.json',dict(persistent_root=str(ROOT),commit=os.environ['PODRUN_COMMIT']))
    for name in ['judge_budget.json']:
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,DEST/name)
    for name in ['training-packages.txt','vllm-packages.txt']:
        if (RUN/name).exists():shutil.copyfile(RUN/name,DEST/name)
    if (RUN/'aggregate.json').exists():
        old=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/C['parent_experiment'];old.mkdir(parents=True,exist_ok=True)
        atomic(old/'direct_student_results.json',aggregate())
        env=dict(os.environ,AUDITBENCH_SHARED=C['parent_root'],PODRUN_REPORT_DIR=str(old),AUDITBENCH_DIRECT_RESULTS=str(old/'direct_student_results.json'))
        subprocess.run([sys.executable,str(HERE.parent/C['parent_experiment']/'report.py')],env=env,check=True)
    subprocess.run(['sync','-f',str(ROOT)],check=True)
    subprocess.run(['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'],check=True)

def publish():
    sys.path.insert(0,str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    DEST.mkdir(parents=True,exist_ok=True);atomic(DEST/'snapshot.json',snapshot(RUN,str(REPORT)))
if __name__=='__main__':{'save':save,'publish':publish}[sys.argv[1]]()
