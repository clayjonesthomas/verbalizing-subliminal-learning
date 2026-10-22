"""Read-only local report watcher; lifecycle is owned by the pod-local cleanup guard."""
import fcntl,hashlib,json,subprocess,sys,time,shlex,os
from pathlib import Path
from report import render
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client,ssh_args
REMOTE='/workspace/projects/verbalizing-subliminal-learning/psychosis-paired-soft/repo/fork_experiments/'+HERE.name
COLLECT='''import json,time,subprocess
from pathlib import Path
p=Path(REMOTE)/'results'
def read(n,default={}):
 try:return json.loads((p/n).read_text())
 except (FileNotFoundError,ValueError):return default
def rows(n):
 out=[]
 try:
  for l in (p/n).read_text().splitlines():
   try:out.append(json.loads(l))
   except ValueError:pass
 except FileNotFoundError:pass
 return out
d={'time':time.time()}
for name in ['progress','supervisor','manifest','dataset','baselines','selection','objective_check','outcome','failure','cleanup','cleanup_environment','lifecycle']:d[name]=read(name+'.json')
for k in [128,256]:
 d['train'+str(k)]=rows(f'k{k}/training.jsonl');d['eval'+str(k)]=rows(f'k{k}/evaluations.jsonl')
d['gpu']=subprocess.getoutput('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader')
print(json.dumps(d))
'''.replace('REMOTE',repr(REMOTE))
def main():
    out=HERE/'results';out.mkdir(exist_ok=True)
    lock=(out/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'watch.pid').write_text(str(os.getpid()))
    c=Client();pod=c.pod('i2p4nlv8scafb7');data={}
    while True:
        try:
            result=subprocess.run(ssh_args(pod,['python','-c',COLLECT]),capture_output=True,text=True,check=True,timeout=90)
            data=json.loads(result.stdout)
            p=out/'report_snapshot.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data));t.replace(p)
            render(data)
            print(time.strftime('%H:%M:%S'),data.get('progress'),flush=True)
            if data.get('supervisor',{}).get('status') in ('success','failed'):
                render(data,'Job finished; pod-local cleanup will terminate the pod. Results persist on the network volume.')
                return
        except Exception as e:
            print(type(e).__name__,str(e),flush=True)
            try:
                c.verify()
                resource=c.request('GET','pods/i2p4nlv8scafb7')
                if resource.get('status') in ('TERMINATED','EXITED'):
                    render(data,'Pod no longer running. Results persist on network volume; local snapshot may be incomplete.');return
            except Exception:
                pass
            render(data,'SSH unavailable; pod-local cleanup remains independent of this laptop. Last sample retained.')
        time.sleep(30)
if __name__=='__main__':main()
