"""Independent tmux guardian: retire if the supervisor vanishes or exceeds 17.5h."""
import base64,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
POD=sys.argv[1];secrets=json.load(sys.stdin)
headers={'Authorization':'Bearer '+secrets['runpod'],'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'}
def api(q):
 req=urllib.request.Request('https://api.runpod.io/graphql',headers=headers,data=json.dumps({'query':q}).encode())
 with urllib.request.urlopen(req,timeout=45) as f:d=json.load(f)
 assert not d.get('errors');return d['data']
def verify():
 d=api('{myself{id} pod(input:{podId:'+json.dumps(POD)+'}){id networkVolume{id}}}')
 assert d['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
 if d['pod']:assert d['pod']['networkVolume']['id']=='ae6w0yoyy6'
 return d['pod']
def write(name,d):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**d),indent=2));t.replace(p)
identity=next((v.split(b'=',1)[1].decode() for v in Path('/proc/1/environ').read_bytes().split(b'\0') if v.startswith(b'RUNPOD_POD_ID=')),None);assert identity==POD
verify();deadline=time.time()+17.5*3600
while True:
 p=OUT/'supervisor_pid.json'
 alive=False
 if p.exists():
  pid=json.loads(p.read_text())['pid'];cmd=Path('/proc')/str(pid)/'cmdline'
  alive=cmd.exists() and b'supervise.py' in cmd.read_bytes()
 if not alive or time.time()>deadline:break
 write('guardian.json',dict(stage='watching',supervisor_pid=pid,pod=POD));time.sleep(30)
if verify() is None:sys.exit(0)
for name in ['training_pid.json','supervisor_pid.json']:
 try:
  pid=json.loads((OUT/name).read_text())['pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
  if str(HERE).encode() in cmd:os.kill(pid,signal.SIGKILL)
 except (OSError,ValueError):pass
write('failure.json',dict(type='SupervisorDisappearedOrTimedOut',message='Independent guardian detected supervisor exit or hard timeout',pod=POD))
try:
 from report import render
 render(HERE)
 subprocess.run(['git','add','-f',str(HERE.relative_to(ROOT)/'results'),str(HERE.relative_to(ROOT)/'report.html')],cwd=ROOT,check=True,timeout=120)
 subprocess.run(['git','-c','user.name=Experiment recorder','-c','user.email=experiment@localhost','commit','-m','Preserve interrupted psychosis continuation artifacts'],cwd=ROOT,check=True,timeout=120)
 auth=base64.b64encode(('x-access-token:'+secrets['github']).encode()).decode();env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+auth)
 p=subprocess.run(['git','push','https://github.com/clayjonesthomas/verbalizing-subliminal-learning.git','HEAD:refs/heads/experiments/psychosis-salve-5k-2026-10-22'],cwd=ROOT,env=env,capture_output=True,timeout=180)
 write('guardian_publication.json',dict(pushed=p.returncode==0))
except Exception:write('guardian_publication.json',dict(pushed=False))
try:subprocess.run(['sync','-f',str(OUT)],timeout=30)
except Exception:pass
while True:
 try:
  if verify() is None:break
  write('retirement.json',dict(status='requested_by_independent_guardian',pod=POD))
  api('mutation {podTerminate(input:{podId:'+json.dumps(POD)+'})}');break
 except Exception:time.sleep(30)
