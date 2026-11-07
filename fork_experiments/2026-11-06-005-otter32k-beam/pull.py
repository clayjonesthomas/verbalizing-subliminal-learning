"""Recover the managed run's complete publication from main or its flagged review branch."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
c=json.loads(Path(sys.argv[1]).read_text());run=c['run_id'];relative=f'fork_experiments/{HERE.name}/results/{run}/snapshot.json'
subprocess.run(['git','fetch','--no-tags','origin'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for ref in ['origin/main','origin/review/'+run]:
 result=subprocess.run(['git','show',ref+':'+relative],cwd=ROOT,capture_output=True)
 if result.returncode==0:
  sys.stdout.buffer.write(result.stdout);break
else:raise SystemExit('Final snapshot not yet published')
