"""Laptop-side result synchronization/report refresh and successful pod retirement."""
import argparse,json,time,subprocess,sys,shlex,traceback,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client
from report import render
REMOTE='/workspace/projects/verbalizing-subliminal-learning/psychosis-salve-convergence/repo/'+str(HERE.relative_to(ROOT))

def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)

def main():
 a=argparse.ArgumentParser();a.add_argument('--publish',type=Path,required=True);args=a.parse_args()
 out=HERE/'results';out.mkdir(exist_ok=True)
 lock=(out/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pid=json.loads((out/'pod.json').read_text())['id'];c=Client();c.verify()
 args.publish.mkdir(parents=True,exist_ok=True)
 for name in ['config.json']:
  (args.publish/name).write_bytes((HERE/name).read_bytes())
 def publish():
  subprocess.run(['rsync','-a','--exclude=*.pt','--exclude=*.log',str(out)+'/',str(args.publish/'results')+'/'],check=True)
  render(HERE);render(args.publish)
 while True:
  try:
   c.verify()
   p=c.request('POST','graphql',{'query':'{ pod(input: {podId: '+json.dumps(pid)+'}) { id desiredStatus networkVolume { id } runtime { ports { ip isIpPublic privatePort publicPort type } } } }'})['data']['pod']
   if not p:
    write(out/'sync.json',dict(time=time.time(),error='Pod absent; no further remote updates'));publish();return
   assert p['id']==pid and p['networkVolume']['id']=='ae6w0yoyy6'
   ports=[x for x in (p.get('runtime') or {}).get('ports',[]) if x['isIpPublic'] and x['privatePort']==22 and x['type']=='tcp']
   if not ports:raise RuntimeError('Pod has no direct SSH endpoint')
   port=ports[0];ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(Path('~/.cache/runpod-vsl/known_hosts').expanduser()),'-o','ConnectTimeout=15','-p',str(port['publicPort'])]
   subprocess.run(['rsync','-az','--exclude=*.pt','--exclude=watch*','--exclude=sync.json','--exclude=pod.json','-e',shlex.join(ssh),'root@'+port['ip']+':'+REMOTE+'/results/',str(out)+'/'],check=True,timeout=90)
   write(out/'sync.json',dict(time=time.time(),pod=pid));publish()
   lifecycle=json.loads((out/'lifecycle.json').read_text()) if (out/'lifecycle.json').exists() else {}
   # Successful-run termination is owned by cleanup_guardian.py on the pod.
   if lifecycle.get('stage')=='failed_pod_retained':return
  except Exception as e:
   write(out/'sync.json',dict(time=time.time(),error=str(e)));publish()
   if (out/'retirement.json').exists():return
  time.sleep(30)
if __name__=='__main__':main()
