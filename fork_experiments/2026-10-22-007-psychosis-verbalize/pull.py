import sys,subprocess,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'setup'))
from runpod import Client,ssh_args,VOLUME
c=Client();c.verify();pods=c.request('GET','pods');pods=pods if isinstance(pods,list) else pods.get('pods',[])
for p in pods:
 if p.get('status')=='RUNNING' and any(m.get('volumeId')==VOLUME for m in p.get('mounts',{}).get('network',[])):
  r=subprocess.run(ssh_args(c.pod(p['id']),['cat','/workspace/projects/verbalizing-subliminal-learning/fork_experiments/2026-10-22-007-psychosis-verbalize/results/final/durable-snapshot.json']),capture_output=True,text=True)
  if r.returncode==0:print(r.stdout);break
else:raise SystemExit('Final snapshot preserved on network volume; no running pod is available for recovery')
