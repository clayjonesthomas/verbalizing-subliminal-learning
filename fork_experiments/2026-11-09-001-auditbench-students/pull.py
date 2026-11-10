"""Git fallback fetches reports only; raw runtime snapshots stay on the volume."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
config = json.loads(Path(sys.argv[1]).read_text())
subprocess.run(['git', 'fetch', '--no-tags', 'origin'], cwd=ROOT, check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
payload = {}
for name in ['in_progress.html', 'final_report.html', 'methods.html']:
    page = subprocess.run(['git', 'show', 'origin/main:fork_experiments/'+HERE.name+'/'+name],
                          cwd=ROOT, capture_output=True)
    if page.returncode == 0:
        payload[name] = page.stdout.decode()
if not payload:
    raise SystemExit('Published reports not yet available')
print(json.dumps(payload))
