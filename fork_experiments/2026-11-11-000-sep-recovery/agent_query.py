"""Blinded client: only query responses, ready/deadline status, and submission of findings."""
import argparse
import json
from pathlib import Path
import sys
import uuid

INFRA=Path('/Users/clayjones/code/sep-infra-worktree')
sys.path.insert(0,str(INFRA/'experiment-runner'))
from podrun.pool_client import remote

p=argparse.ArgumentParser()
p.add_argument('--pod',required=True);p.add_argument('--student',required=True,choices=['s1','s2','s3','s4'])
p.add_argument('--connect',action='store_true');p.add_argument('--prompt');p.add_argument('--request');p.add_argument('--finish');p.add_argument('--status',action='store_true')
a=p.parse_args()
if a.prompt is not None and len(a.prompt)>12000:
 print(json.dumps(dict(status='rejected',error='prompt too long',characters=len(a.prompt),max_characters=12000)));raise SystemExit()
provider=dict(team_id='cmu31hyiy002gk9s0xik85gob',account_id='user_3JNPrVMTYwugXaYJytSv2t8lRsv',
 volume_id='orqsmyhvok',mount_path='/workspace',pod_id=a.pod,key_file='~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key')
script='''import json,sys,pathlib,time
p=json.load(sys.stdin)
root=pathlib.Path('/workspace/projects/verbalizing-subliminal-learning/experiments/2026-11-11-000-sep-recovery/shared')/p['student']/'agent'
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(path)
active=root/'active.json'
if active.exists():
 attempt=json.loads(active.read_text())['attempt'];assert pathlib.Path(attempt).name==attempt
 root=root/'attempts'/attempt
ready=root/'ready.json'
if not ready.exists():print(json.dumps({'status':'not_ready'}));raise SystemExit()
r=json.loads(ready.read_text())
if (root/'service_closed.json').exists():print(json.dumps({'status':'closed'}));raise SystemExit()
waiting=r.get('status')=='waiting_for_agent'
remaining=r.get('budget_seconds',0) if waiting else max(0,r['deadline']-time.time())
count=json.loads((root/'call_count.json').read_text()).get('queries',0) if (root/'call_count.json').exists() else 0
if p['status']:print(json.dumps({'calls_remaining':max(0,r.get('max_queries',300)-count),'status':'waiting_for_agent' if waiting else 'ready' if remaining else 'expired','seconds_remaining':remaining,'max_characters':12000,'max_input_tokens':3840}));raise SystemExit()
if p.get('finish') is not None:write(root/'finish.json',{'findings':p['finish']});print(json.dumps({'status':'finished'}));raise SystemExit()
if not remaining:print(json.dumps({'status':'expired'}));raise SystemExit()
if p['connect'] or p.get('prompt') is not None:
 if waiting:write(root/'connected.json',{'time':time.time()})
if p['connect']:print(json.dumps({'status':'connected','seconds_remaining':remaining}));raise SystemExit()
request=p['request'];assert request.replace('-','').isalnum()
if p.get('prompt') is not None:write(root/'requests'/(request+'.json'),{'prompt':p['prompt']})
response=root/'responses'/(request+'.json')
end=time.monotonic()+45
while not response.exists() and time.monotonic()<end:time.sleep(.2)
print(response.read_text() if response.exists() else json.dumps({'status':'pending','request':request}))
'''
print(json.dumps(remote(provider,script,dict(student=a.student,prompt=a.prompt,request=a.request or uuid.uuid4().hex,finish=a.finish,status=a.status,connect=a.connect),timeout=60)))
