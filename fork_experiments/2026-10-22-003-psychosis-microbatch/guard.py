"""Independent cleanup guard: artifact-write failures must not prevent termination."""
import sys,json,time,os,urllib.request
from pathlib import Path
pod,pid,out,commit=sys.argv[1:];pid=int(pid);out=Path(out);key=sys.stdin.readline().strip();assert key
TEAM='cmu31hyiy002gk9s0xik85gob';ACCOUNT='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
def api(path,body):
 url='https://api.runpod.io/graphql' if path=='graphql' else 'https://api.runpod.io/v2/'+path
 req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'X-Team-Id':TEAM,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:
  b=r.read();return json.loads(b) if b else {}
def verify():
 assert api('graphql',{'query':'{myself{id}}'})['data']['myself']['id']==ACCOUNT
 actual=next(x.split(b'=',1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\x00') if x.startswith(b'RUNPOD_POD_ID='));assert actual==pod
verify()
def identity():
 try:
  s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return None if s[0]=='Z' else s[19]
 except FileNotFoundError:return None
original=identity();assert original
assert str(out.parent/'supervise.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
def record(status):
 try:(out/'guard.json').write_text(json.dumps(dict(status=status,pod=pod,pid=os.getpid(),supervisor_pid=pid,source_commit=commit,time=time.time())))
 except OSError:pass
record('armed')
while identity()==original:time.sleep(15)
record('supervisor_exited_terminating')
# A failed archive/log write must never bypass API cleanup.
while True:
 try:verify();api('pods/'+pod+'/action',{'action':'terminate'});break
 except Exception:time.sleep(20)
