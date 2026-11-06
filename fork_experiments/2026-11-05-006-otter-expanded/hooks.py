"""Save scientific artifacts and publish durable recovery; shared runner owns cleanup."""
import gzip
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ['PODRUN_DIR'])
REPORT = Path(os.environ.get('PODRUN_REPORT_DIR', HERE))
REPORTS = ['in_progress.html', 'final_report.html', 'methods.html']


def save():
    subprocess.run(['bash', '/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'], check=True)
    from report import render
    render()
    for n in REPORTS:
        shutil.copyfile(REPORT / n, OUT / n)
    for n in ['uv.lock', 'pyproject.toml']:
        shutil.copyfile(ROOT / n, OUT / n)
    if not (OUT/'scientific_config.json').exists():
        shutil.copyfile(HERE / 'config.json', OUT / 'scientific_config.json')
    shutil.copyfile(HERE / 'requirements.txt', OUT / 'requirements.txt')
    (OUT / 'environment-packages.txt').write_text('\n'.join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + '\n')
    checkpoints = [p for p in (OUT/'checkpoints').rglob('*') if p.is_file()] if (OUT/'checkpoints').exists() else []
    (OUT/'checkpoint_manifest.json').write_text(json.dumps({str(p.relative_to(OUT)): dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in checkpoints},indent=2))
    files = [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json', '.jsonl', '.html', '.txt']
             and 'publication' not in p.parts and p.name != 'artifact_hashes.json']
    (OUT / 'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
    subprocess.run(['sync', '-f', str(OUT)], check=True)


def publish():
    sys.path.insert(0, str(OUT / 'runner.pyz'))
    from podrun.watcher import snapshot
    pub = OUT / 'publication'
    pub.mkdir(exist_ok=True)
    (pub / 'snapshot.json').write_text(json.dumps(snapshot(OUT, REPORT)))
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.json', '.jsonl', '.html', '.txt', '.toml', '.lock'] and p.name != 'snapshot.json':
            shutil.copyfile(p, pub / p.name)
    for name in ['generations', 'scores']:
        if (OUT / name).exists():
            shutil.copytree(OUT / name, pub / name, dirs_exist_ok=True)
    for p in OUT.glob('*.log'):
        with p.open('rb') as source, gzip.open(pub / (p.name + '.gz'), 'wb') as target:
            shutil.copyfileobj(source, target)
    git = ['git', '-C', str(pub)]
    subprocess.run(['git', 'init', str(pub)], check=True)
    ssh = subprocess.check_output(['git', '-C', '/workspace/projects/verbalizing-subliminal-learning/repo', 'config', 'core.sshCommand'], text=True).strip()
    assert 'github_known_hosts' in ssh
    subprocess.run(git + ['config', 'core.sshCommand', ssh], check=True)
    subprocess.run(git + ['add', '.'], check=True)
    subprocess.run(git + ['-c', 'user.name=Experiment recorder', '-c', 'user.email=experiment@localhost',
                         'commit', '--allow-empty', '-m', 'Save otter training results'], check=True)
    branch = 'artifacts/' + os.environ['PODRUN_ID']
    subprocess.run(git + ['push', 'git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git', 'HEAD:refs/heads/' + branch], check=True)
    (OUT / 'published.json').write_text(json.dumps(dict(branch=branch, commit=subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip())))


if __name__ == '__main__':
    {'save': save, 'publish': publish}[sys.argv[1]]()
