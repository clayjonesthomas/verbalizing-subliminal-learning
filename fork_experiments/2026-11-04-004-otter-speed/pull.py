"""Recover a final artifact commit for the shared experiment watcher."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
config = json.loads(Path(sys.argv[1]).read_text())
branch = 'artifacts/' + config['run_id']
ref = 'refs/remotes/origin/' + branch
subprocess.run(['git', 'fetch', '--no-tags', '--no-write-fetch-head', 'origin', 'refs/heads/' + branch + ':' + ref],
               cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
commit = subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
dest = HERE / 'results/recovered'
for name in subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=ROOT, text=True).splitlines():
    p = Path(name)
    assert not p.is_absolute() and '..' not in p.parts
    value = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)
    out = dest / p
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(value)
    if name in ['in_progress.html', 'final_report.html', 'methods.html']:
        (HERE / name).write_bytes(value)
(dest / 'recovery.json').write_text(json.dumps(dict(branch=branch, commit=commit)))
sys.stdout.buffer.write((dest / 'snapshot.json').read_bytes())
