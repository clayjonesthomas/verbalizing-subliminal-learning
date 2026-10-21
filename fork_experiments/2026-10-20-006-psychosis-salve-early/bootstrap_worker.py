"""Receive ephemeral credentials and detach the worker guard after CUDA validation."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--role',choices=['student','salve'],required=True);p.add_argument('--pod',required=True);a=p.parse_args()
import torch
assert torch.cuda.is_available() and torch.ones(1,device='cuda').item()==1
secrets=json.load(sys.stdin);secret=Path('/run/psychosis-secrets');secret.mkdir(mode=0o700,exist_ok=True)
for name,value in [('runpod',secrets['runpod']),('OPENROUTER_API_KEY',secrets['openrouter'])]:
 f=secret/name;f.write_text(value);f.chmod(0o600)
here=Path(__file__).resolve().parent;out=here/'results/workers'/a.role;out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PSYCHOSIS_ROLE=a.role,OPENROUTER_API_KEY=secrets['openrouter'],SSH_AUTH_SOCK='',PYTHONUNBUFFERED='1')
with (out/'supervisor.log').open('a') as log:
 child=subprocess.Popen([sys.executable,str(here/'guard.py'),'--pod',a.pod,'--keyfile',str(secret/'runpod')],cwd=here.parents[1],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'role':a.role,'guard_pid':child.pid,'cuda_ready':True}))
