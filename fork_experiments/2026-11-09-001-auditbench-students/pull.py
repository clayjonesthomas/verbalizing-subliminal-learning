import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
config=json.loads(Path(sys.argv[1]).read_text());run=config['run_id']
subprocess.run(['git','fetch','--no-tags','origin'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for ref in ['origin/main','origin/review/'+run]:
    result=subprocess.run(['git','show',ref+f':fork_experiments/{HERE.name}/results/{run}/snapshot.json'],cwd=ROOT,capture_output=True)
    if result.returncode==0:
        payload=json.loads(result.stdout)
        for name in ['in_progress.html','final_report.html','methods.html']:
            page=subprocess.run(['git','show',ref+f':fork_experiments/{HERE.name}/'+name],cwd=ROOT,capture_output=True)
            if page.returncode==0:payload[name]=page.stdout.decode()
        print(json.dumps(payload));break
else:raise SystemExit('Published snapshot not yet available')
