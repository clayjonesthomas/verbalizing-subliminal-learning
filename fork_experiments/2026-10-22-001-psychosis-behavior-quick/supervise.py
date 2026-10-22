"""Pod-local completion/failure cleanup, independent of the laptop."""
import json,sys,time,subprocess,urllib.request,os,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ROOT=HERE.parents[1]
def write(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**data),indent=2));t.replace(p)
def main():
 pod=sys.argv[1];secrets=json.load(sys.stdin)
 identity=next((x.split(b'=',1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\0') if x.startswith(b'RUNPOD_POD_ID=')),None);assert pod==identity
 headers={'Authorization':'Bearer '+secrets['runpod'].strip(),'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'}
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
 OUT.mkdir(exist_ok=True)
 try:
  verify();write('lifecycle.json',dict(stage='running',pod=pod))
  snapshot=Path('/workspace/projects/verbalizing-subliminal-learning/venv-snapshot.tar');assert snapshot.exists()
  write('environment_snapshot.json',dict(path=str(snapshot),bytes=snapshot.stat().st_size,note='Existing persisted environment; exact runtime package versions in manifest.json'))
  with (OUT/'run.log').open('a') as log:
   subprocess.run([sys.executable,str(HERE/'run.py')],cwd=ROOT,env=dict(os.environ,OPENROUTER_API_KEY=secrets['openrouter']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
 except BaseException:
  write('failure.json',dict(traceback=traceback.format_exc()));write('progress.json',dict(stage='failed'))
 finally:
  try:
   from report import render
   render();subprocess.run(['sync','-f',str(OUT)],check=True,timeout=30)
  except Exception:pass
  write('lifecycle.json',dict(stage='artifacts_saved_retiring_in_60_seconds',pod=pod))
  time.sleep(60)
  while True:
   try:
    if verify() is None:return
    write('retirement.json',dict(status='requested',pod=pod))
    api('mutation {podTerminate(input:{podId:'+json.dumps(pod)+'})}')
    write('retirement.json',dict(status='accepted',pod=pod));return
   except Exception:time.sleep(30)
if __name__=='__main__':main()
