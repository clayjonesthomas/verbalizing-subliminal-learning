"""Persist artifacts and publish recovery; pod lifecycle belongs to podrun."""
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=Path(os.environ['PODRUN_DIR'])
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
PATTERNS=['salve/*/seed*/result.json','*/greedy/seed*/result.json','prompt_fluency.json','salve/*/seed*/training_history.json']


def save():
    files={str(p.relative_to(OUT)):json.loads(p.read_text()) for pattern in PATTERNS for p in OUT.glob(pattern)}
    (OUT/'report_data.json').write_text(json.dumps(files))
    (OUT/'environment-packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')
    for name in ['uv.lock','pyproject.toml']:shutil.copyfile(ROOT/name,OUT/name)
    shutil.copyfile(HERE/'config.json',OUT/'scientific_config.json')
    checkpoints={}
    for p in OUT.glob('salve/*/seed*/soft_prompt.pt'):
        with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        checkpoints[str(p)]=dict(sha256=digest,bytes=p.stat().st_size)
    (OUT/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2))
    manifest=OUT/'manifest.json'
    if manifest.exists():
        from report import render
        render(group_root=Path(json.loads(manifest.read_text())['group_root']))
    subprocess.run(['sync','-f',str(OUT)],check=True)


def publish():
    sys.path.insert(0,str(OUT/'runner.pyz'))
    from podrun.watcher import snapshot
    pub=OUT/'publication';pub.mkdir(exist_ok=True)
    data=snapshot(OUT,REPORT)
    data['report_data.json']=(OUT/'report_data.json').read_text()
    (pub/'snapshot.json').write_text(json.dumps(data))
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.json','.jsonl','.txt','.lock','.toml']:
            shutil.copyfile(p,pub/p.name)
    for p in OUT.glob('*.log'):
        with p.open('rb') as src,gzip.open(pub/(p.name+'.gz'),'wb') as dst:shutil.copyfileobj(src,dst)
    for name in ['in_progress.html','final_report.html','methods.html']:
        if (REPORT/name).exists():shutil.copyfile(REPORT/name,pub/name)
    git=['git','-C',str(pub)]
    subprocess.run(['git','init',str(pub)],check=True)
    ssh=subprocess.check_output(['git','-C','/workspace/projects/verbalizing-subliminal-learning/repo','config','core.sshCommand'],text=True).strip()
    assert 'github_known_hosts' in ssh
    subprocess.run(git+['config','core.sshCommand',ssh],check=True)
    subprocess.run(git+['add','.'],check=True)
    subprocess.run(git+['-c','user.name=Experiment recorder','-c','user.email=experiment@localhost','commit','--allow-empty','-m','Preserve Figure 4 expansion worker results'],check=True)
    branch='artifacts/'+os.environ['PODRUN_ID']
    subprocess.run(git+['push','git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git','HEAD:refs/heads/'+branch],check=True)
    (OUT/'published.json').write_text(json.dumps({'branch':branch,'commit':subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()}))


if __name__=='__main__':
    {'save':save,'publish':publish}[sys.argv[1]]()
