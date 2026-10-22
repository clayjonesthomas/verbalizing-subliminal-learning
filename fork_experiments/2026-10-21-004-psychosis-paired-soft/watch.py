"""Poll read-only metrics; download checkpoints before retiring a successful pod."""
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
for name in ['progress','supervisor','manifest','dataset','baselines','selection','objective_check','outcome','failure']:d[name]=read(name+'.json')
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
            if data.get('supervisor',{}).get('status')=='success':
                args=ssh_args(pod,['true']);host=args[-2];transport=shlex.join(args[:args.index(host)])
                subprocess.run(['rsync','-az','-e',transport,host+':'+REMOTE+'/results/',str(out)+'/'],check=True,timeout=300)
                check="import json,hashlib;from pathlib import Path;p=Path("+repr(REMOTE)+")/'results';print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}))"
                result=subprocess.run(ssh_args(pod,['python','-c',check]),capture_output=True,text=True,check=True,timeout=90)
                hashes=json.loads(result.stdout)
                for name,digest in hashes.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
                c.pod('i2p4nlv8scafb7') # reverify team + volume
                c.request('POST','pods/i2p4nlv8scafb7/action',{'action':'terminate'})
                (out/'lifecycle.json').write_text(json.dumps(dict(status='complete',terminated=True,time=time.time(),hashes=hashes)))
                render(data,'Complete; checkpoints downloaded and verified; pod terminated; network volume preserved')
                return
            if data.get('supervisor',{}).get('status')=='failed':
                render(data,'Failed; pod preserved for inspection');return
        except Exception as e:
            print(type(e).__name__,str(e),flush=True);render(data,'Connection/finalization error: '+str(e))
        time.sleep(30)
if __name__=='__main__':main()
