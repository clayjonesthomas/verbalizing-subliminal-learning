"""Recover the published snapshot through any live pod on the same project volume."""
import sys,json,subprocess
from pathlib import Path
h=Path(__file__).resolve().parent;sys.path.insert(0,str(h.parents[1]/'setup'))
from runpod import Client,ssh_args,VOLUME
c=Client();c.verify();pods=c.request('GET','pods');pods=pods if isinstance(pods,list) else pods.get('pods',[])
config=json.loads((h/'config.json').read_text());target=config['shared_root']+'/results/'+sys.argv[1]+'/durable-snapshot.json'
for p in pods:
 if p.get('status')=='RUNNING' and any(m.get('volumeId')==VOLUME for m in p.get('mounts',{}).get('network',[])):
  r=subprocess.run(ssh_args(c.pod(p['id']),['cat',target]),capture_output=True,text=True)
  if r.returncode==0:print(r.stdout);break
else:raise SystemExit('Snapshot preserved on network volume; recovery needs a running pod on that volume')
