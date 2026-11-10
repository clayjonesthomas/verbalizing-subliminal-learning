"""Recover published scoring results and update the existing mega report."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
run=sys.argv[1]
subprocess.run(['git','fetch','--no-tags','origin'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for ref in ['origin/main','origin/review/'+run]:
    prefix=ref+f':fork_experiments/{HERE.name}/results/{run}/'
    result=subprocess.run(['git','show',prefix+'snapshot.json'],cwd=ROOT,capture_output=True)
    if result.returncode:continue
    summary=subprocess.run(['git','show',prefix+'summary.json'],cwd=ROOT,capture_output=True,check=True)
    data=json.loads(summary.stdout)
    if data['status']=='complete' and len(data['results'])==28 and all(r['matches_previous'] for r in data['results']):
        folder=HERE.parent/'2026-11-09-002-dpo-mega';dest=folder/'measurements/loss_intervals.json'
        if not dest.exists() or json.loads(dest.read_text())!=data:
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_bytes(summary.stdout);tmp.replace(dest)
            subprocess.run([sys.executable,str(folder/'report.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    print(json.dumps(json.loads(result.stdout)));break
else:raise SystemExit('Published snapshot not yet available')
