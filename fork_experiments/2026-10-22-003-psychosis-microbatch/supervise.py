"""Detached bounded training with pod-local reporting and cleanup on either outcome."""
import json,os,subprocess,sys,time,urllib.request,traceback,threading,signal,shutil,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';OUT.mkdir(exist_ok=True)
pod=sys.argv[1];key=sys.stdin.readline().strip();assert key
C=json.loads((HERE/'config.json').read_text())
def write(n,d):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def api(path,body):
 url='https://api.runpod.io/graphql' if path=='graphql' else 'https://api.runpod.io/v2/'+path
 req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:
  b=r.read();return json.loads(b) if b else {}
def verify():
 assert api('graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
 actual=next(x.split(b'=',1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\x00') if x.startswith(b'RUNPOD_POD_ID='))
 assert actual==pod
verify()
write('supervisor.json',dict(pid=os.getpid(),pod=pod,status='starting',time=time.time()))
stop=threading.Event()
def report_loop():
 while not stop.is_set():
  try:
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader'],text=True,timeout=15).strip()
   with (OUT/'telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu=gpu))+'\n')
   subprocess.run([sys.executable,str(HERE/'report.py')],timeout=20,check=True)
  except Exception as e:print('Reporting:',type(e).__name__,flush=True)
  stop.wait(30)
def handoff():
 parent=Path(C['parent_experiment']);old=json.loads((parent/'results/supervisor.json').read_text());pid=old['pid']
 assert old['status']=='running' and old['pod']==pod
 assert str(parent/'supervise.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
 assert os.getpgid(pid)==pid
 # Stop the original process group before snapshotting its atomic checkpoints.
 os.killpg(pid,signal.SIGSTOP)
 children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
 assert any(str(parent/'run.py').encode() in Path(f'/proc/{child}/cmdline').read_bytes() for child in children)
 (HERE/'assets').mkdir(exist_ok=True)
 shutil.copytree(parent/'assets/dataset',HERE/'assets/dataset',copy_function=os.link)
 for f in (parent/'results').iterdir():
  if f.is_dir() and f.name.startswith('k'):
   shutil.copytree(f,OUT/f.name,ignore=shutil.ignore_patterns('*.tmp'))
  elif f.name in ('baselines.json','base_train.json','base_valid.json','dataset.json','context_example.json','selection.json','selection_history.jsonl','batch_order.json'):
   shutil.copyfile(f,OUT/f.name)
 shutil.copyfile(parent/'results/manifest.json',OUT/'parent_manifest.json')
 checkpoints={str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob('k*/*.pt')}
 write('continuation.json',dict(parent=str(parent),parent_commit=json.loads((OUT/'parent_manifest.json').read_text())['source_commit'],checkpoint_sha256=checkpoints,time=time.time(),policy='Resume latest atomic checkpoints; replay any updates after last checkpoint; preserve original data order and wall deadline'))
 # The new supervisor now owns cleanup; the old supervisor must not run its finally block.
 os.killpg(pid,signal.SIGKILL)
 (parent/'results/handoff.json').write_text(json.dumps(dict(continuation=str(HERE),time=time.time(),reason='user-authorized microbatch continuation')))
status='failed'
try:
 if not (OUT/'continuation.json').exists():handoff()
 subprocess.run(['uv','pip','install','--python',sys.executable,'--target','/opt/psychosis-system-soft-packages','--no-deps','flash-linear-attention==0.5.2','fla-core==0.5.2','einops==0.8.1','triton==3.7.1'],check=True,timeout=180)
 env=dict(os.environ,PYTHONPATH='/opt/psychosis-system-soft-packages')
 subprocess.run([sys.executable,'-m','unittest','test_data.py'],cwd=HERE,env=env,check=True,timeout=60)
 write('supervisor.json',dict(pid=os.getpid(),pod=pod,status='running',time=time.time(),terminate_on='success or failure',wall_cap_hours=C['max_wall_hours']))
 thread=threading.Thread(target=report_loop,daemon=True);thread.start()
 subprocess.run([sys.executable,'-u',str(HERE/'run.py')],env=env,cwd=HERE,check=True,timeout=(C['max_wall_hours']+0.25)*3600)
 status='success'
except BaseException:
 write('supervisor_failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
 traceback.print_exc()
finally:
 stop.set()
 write('supervisor.json',dict(pid=os.getpid(),pod=pod,status=status,time=time.time()))
 try:
  subprocess.run(['tar','-cf',str(HERE/'assets/environment.tar'),'-C','/','opt/psychosis-system-soft-packages','opt/verbalizing-subliminal-learning-venv'],check=True,timeout=600)
 except Exception as e:write('environment_error.json',dict(type=type(e).__name__))
 write('lifecycle.json',dict(status=status,pod=pod,termination_requested=True,time=time.time(),artifacts='network volume preserved'))
 subprocess.run([sys.executable,str(HERE/'report.py')],timeout=30)
 subprocess.run(['sync','-f',str(OUT)],timeout=30)
 while True:
  try:
   verify();api('pods/'+pod+'/action',{'action':'terminate'});break
  except Exception:time.sleep(20)
