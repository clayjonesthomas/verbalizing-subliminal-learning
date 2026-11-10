import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
run=sys.argv[1]
subprocess.run(['git','fetch','--no-tags','origin'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for ref in ['origin/main','origin/review/'+run]:
    result=subprocess.run(['git','show',ref+f':fork_experiments/{HERE.name}/results/{run}/snapshot.json'],cwd=ROOT,capture_output=True)
    if result.returncode==0:
        print(json.dumps(json.loads(result.stdout)));break
else:raise SystemExit('Published snapshot not yet available')
