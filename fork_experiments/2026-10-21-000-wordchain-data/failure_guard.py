"""User-authorized failure termination for an already-running generation job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def write(name,data):
    path=OUT/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def alive(pid):
    try:
        raw=Path(f'/proc/{pid}/cmdline').read_bytes()
        return str(HERE/'supervise.py').encode() in raw
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--pod',required=True);p.add_argument('--pid',required=True,type=int);p.add_argument('--source-commit',required=True);a=p.parse_args()
    key=sys.stdin.readline().strip()
    def gql(query):
        req=urllib.request.Request('https://api.runpod.io/graphql',data=json.dumps({'query':query}).encode(),headers={'Authorization':'Bearer '+key,'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=45) as r:result=json.load(r)
        if result.get('errors'):raise RuntimeError('Runpod GraphQL returned errors')
        return result['data']
    def verify():
        assert gql('{ myself { id email } }')['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
    verify()
    assert Path(__file__).read_bytes()==subprocess.check_output(['git','show',a.source_commit+':fork_experiments/'+HERE.name+'/failure_guard.py'],cwd=HERE)
    write('failure_guard.json',dict(status='armed',pid=os.getpid(),supervisor_pid=a.pid,pod=a.pod,source_commit=a.source_commit,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=time.time(),policy='User requested termination on failure; preserve volume and attempt environment archive'))
    while not (OUT/'failure.json').exists() and alive(a.pid):time.sleep(15)
    write('failure_guard_event.json',dict(time=time.time(),reason='failure recorded or supervisor exited',supervisor_alive=alive(a.pid)))
    archive=HERE/'assets/environment.tar'
    if not archive.exists():
        try:
            paths=[s for s in ['opt/wordchain-infer','opt/wordchain-merge-libs','opt/verbalizing-subliminal-learning-venv'] if Path('/'+s).exists()]
            with (OUT/'failure_environment.log').open('a') as log:
                subprocess.run(['tar','-cf',str(archive)+'.failure.tmp','-C','/',*paths],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            Path(str(archive)+'.failure.tmp').replace(archive)
        except Exception as e:write('failure_environment_error.json',dict(error=str(e),time=time.time()))
    subprocess.run(['sync','-f',str(OUT)],timeout=30,check=False)
    # Reconcile the same pod before every mutation; never touch another resource.
    while True:
        try:
            verify()
            pod=gql('{ pod(input: {podId: '+json.dumps(a.pod)+'}) { id desiredStatus } }')['pod']
            if pod is None or pod.get('desiredStatus')=='TERMINATED':return
            assert pod['id']==a.pod
            write('failure_termination.json',dict(status='requested',pod=a.pod,time=time.time()))
            gql('mutation { podTerminate(input: {podId: '+json.dumps(a.pod)+'}) }')
            write('failure_termination.json',dict(status='accepted',pod=a.pod,time=time.time()))
            return
        except Exception as e:
            write('failure_termination.json',dict(status='retrying after reconciliation',error=str(e),pod=a.pod,time=time.time()))
            time.sleep(30)

if __name__=='__main__':main()
