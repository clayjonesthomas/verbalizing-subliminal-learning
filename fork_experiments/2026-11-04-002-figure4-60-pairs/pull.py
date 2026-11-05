"""Durable per-worker snapshot recovery for the managed watcher."""
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
cfg=json.loads(Path(sys.argv[1]).read_text())
branch='artifacts/'+cfg['run_id']
ref='refs/remotes/origin/'+branch
subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head','origin','refs/heads/'+branch+':'+ref],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
commit=subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()
snapshot=subprocess.check_output(['git','show',commit+':snapshot.json'],cwd=ROOT)
data=json.loads(snapshot)
dest=HERE/'results/recovered'/cfg['run_id'];dest.mkdir(parents=True,exist_ok=True)
for name in ['report_data.json','state.json','manifest.json','checkpoints.json','complete.json']:
    if name in data:(dest/name).write_text(data[name])
(dest/'recovery.json').write_text(json.dumps({'branch':branch,'commit':commit}))
sys.stdout.buffer.write(snapshot)
