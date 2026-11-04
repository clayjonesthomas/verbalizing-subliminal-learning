"""Detached pod-local supervisor; save artifacts and terminate on success or failure."""
import json,sys,time,subprocess,urllib.request,os,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ROOT=HERE.parents[1]
def write(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**data),indent=2));t.replace(p)
def main():
 pod=sys.argv[1]
 identity=next((x.split(b'=',1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\0') if x.startswith(b'RUNPOD_POD_ID=')),None);assert pod==identity
 headers={'Authorization':'Bearer '+sys.stdin.read().strip(),'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'}
 def api(query):
  req=urllib.request.Request('https://api.runpod.io/graphql',headers=headers,data=json.dumps({'query':query}).encode())
  with urllib.request.urlopen(req,timeout=45) as r:result=json.load(r)
  if result.get('errors'):raise RuntimeError('GraphQL operation failed')
  return result['data']
 def verify():
  d=api('{myself{id} pod(input:{podId:'+json.dumps(pod)+'}){id networkVolume{id}}}')
  assert d['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
  if d['pod']:assert d['pod']['id']==pod and d['pod']['networkVolume']['id']=='ae6w0yoyy6'
  return d['pod']
 OUT.mkdir(exist_ok=True);verify()
 try:
  write('lifecycle.json',dict(stage='running',pod=pod))
  with (OUT/'run.log').open('a') as log:subprocess.run([sys.executable,str(HERE/'run.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
 except BaseException:write('failure.json',dict(traceback=traceback.format_exc()))
 finally:
  try:
   write('lifecycle.json',dict(stage='saving_environment',pod=pod))
   dest=HERE/'assets/environment.tar';dest.parent.mkdir(exist_ok=True);tmp=dest.with_suffix('.tar.tmp')
   subprocess.run(['tar','-C','/','-cf',str(tmp),'opt/verbalizing-subliminal-learning-venv'],check=True,timeout=240);tmp.replace(dest)
  except Exception:write('environment_export_failure.json',dict(traceback=traceback.format_exc()))
  try:subprocess.run(['sync','-f',str(OUT)],check=True,timeout=30)
  except Exception:pass
  while True:
   try:
    if verify() is None:return
    write('retirement.json',dict(status='requested',pod=pod))
    api('mutation {podTerminate(input:{podId:'+json.dumps(pod)+'})}')
    write('retirement.json',dict(status='accepted',pod=pod));return
   except Exception:time.sleep(30)
if __name__=='__main__':main()
