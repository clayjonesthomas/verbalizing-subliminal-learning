"""Detached training/reporting, Git publication, and success/failure retirement."""
import base64,json,os,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
BRANCH='experiments/psychosis-salve-5k-2026-10-22'
def write(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**obj),indent=2));t.replace(p)
def main():
 pod=sys.argv[1];secrets=json.load(sys.stdin)
 identity=next((v.split(b'=',1)[1].decode() for v in Path('/proc/1/environ').read_bytes().split(b'\0') if v.startswith(b'RUNPOD_POD_ID=')),None)
 assert identity==pod
 headers={'Authorization':'Bearer '+secrets['runpod'],'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'}
 def api(q):
  req=urllib.request.Request('https://api.runpod.io/graphql',headers=headers,data=json.dumps({'query':q}).encode())
  with urllib.request.urlopen(req,timeout=45) as r:d=json.load(r)
  assert not d.get('errors'),'GraphQL request failed'
  return d['data']
 def verify():
  d=api('{myself{id} pod(input:{podId:'+json.dumps(pod)+'}){id networkVolume{id}}}')
  assert d['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
  if d['pod']:assert d['pod']['id']==pod and d['pod']['networkVolume']['id']=='ae6w0yoyy6'
  return d['pod']
 OUT.mkdir(exist_ok=True);child=None
 write('supervisor_pid.json',dict(pid=os.getpid(),pod=pod,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
 from report import render
 try:
  verify();write('pod.json',dict(id=pod));write('lifecycle.json',dict(stage='running'))
  import importlib.metadata
  write('environment.json',dict(packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},snapshot='/workspace/projects/verbalizing-subliminal-learning/venv-snapshot.tar'))
  assert Path('/workspace/projects/verbalizing-subliminal-learning/venv-snapshot.tar').is_file()
  with (OUT/'run.log').open('a') as log:
   child=subprocess.Popen([sys.executable,str(HERE/'run.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   write('training_pid.json',dict(pid=child.pid))
   deadline=time.time()+17*3600
   while child.poll() is None:
    render(HERE)
    if time.time()>deadline:child.kill();child.wait();raise TimeoutError('17-hour supervisor cap')
    time.sleep(20)
   if child.returncode:raise RuntimeError('Training subprocess exited '+str(child.returncode))
  write('lifecycle.json',dict(stage='training_complete'))
 except BaseException:
  if child is not None and child.poll() is None:child.kill();child.wait()
  write('supervisor_failure.json',dict(traceback=traceback.format_exc()));write('lifecycle.json',dict(stage='failed'))
 finally:
  try:
   render(HERE)
   subprocess.run(['sync','-f',str(OUT)],check=True,timeout=30)
   # Persist results in Git on the volume, then publish using an in-memory credential.
   subprocess.run(['git','add','-f',str(HERE.relative_to(ROOT)/'results'),str(HERE.relative_to(ROOT)/'report.html'),str(HERE.relative_to(ROOT)/'assets/resume_latest.pt')],cwd=ROOT,check=True,timeout=120)
   subprocess.run(['git','-c','user.name=Experiment recorder','-c','user.email=experiment@localhost','commit','-m','Save psychosis 5k continuation results and checkpoints'],cwd=ROOT,check=True,timeout=120)
   result_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
   subprocess.run(['git','bundle','create',str(HERE/'results/final_results.bundle'),'HEAD'],cwd=ROOT,check=True,timeout=180)
   auth=base64.b64encode(('x-access-token:'+secrets['github']).encode()).decode()
   env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+auth)
   for attempt in range(3):
    p=subprocess.run(['git','push','https://github.com/clayjonesthomas/verbalizing-subliminal-learning.git','HEAD:refs/heads/'+BRANCH],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    if p.returncode==0:break
    time.sleep(10)
   write('publication.json',dict(commit=result_commit,branch=BRANCH,pushed=p.returncode==0,attempts=attempt+1))
  except BaseException:write('publication_failure.json',dict(traceback=traceback.format_exc()))
  try:subprocess.run(['sync','-f',str(OUT)],check=True,timeout=30)
  except Exception:pass
  write('lifecycle.json',dict(stage='artifacts_saved_retiring',pod=pod))
  time.sleep(60)
  while True:
   try:
    if verify() is None:return
    write('retirement.json',dict(status='requested',pod=pod))
    api('mutation {podTerminate(input:{podId:'+json.dumps(pod)+'})}')
    write('retirement.json',dict(status='termination_accepted',pod=pod));return
   except Exception:time.sleep(30)
if __name__=='__main__':main()
