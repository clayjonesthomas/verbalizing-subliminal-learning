"""Pod-local cleanup for this run; terminate on success OR failure, per user request.

Credential arrives over stdin and lives only in process memory. Training code and
its recorded commit remain unchanged; this lifecycle addition has its own SHA.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
POD='i2p4nlv8scafb7'
TEAM='cmu31hyiy002gk9s0xik85gob'
ACCOUNT='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
VOLUME='ae6w0yoyy6'

def reason(status,alive,now,last_activity,deadline):
    if status in ('success','failed'):return status
    if not alive:return 'supervisor_exited'
    if now>deadline:return 'wall_timeout'
    if now-last_activity>5400:return 'stalled_90_minutes'
    return None

def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--commit',required=True)
    a=parser.parse_args();here=a.experiment;out=here/'results'
    lock=(out/'cleanup.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    key=sys.stdin.readline().strip();assert key
    def read(name):
        try:return json.loads((out/name).read_text())
        except (FileNotFoundError,ValueError):return {}
    def write(name,data):
        p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
    def api(method,path,body=None):
        url='https://api.runpod.io/graphql' if path=='graphql' else 'https://api.runpod.io/v2/'+path
        req=Request(url,method=method,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+key,'X-Team-Id':TEAM,'Content-Type':'application/json'})
        with urlopen(req,timeout=30) as response:
            data=response.read();return json.loads(data) if data else {}
    def verify():
        identity=api('POST','graphql',{'query':'{ myself { id email } }'})
        assert identity.get('data',{}).get('myself',{}).get('id')==ACCOUNT
        pod=api('GET','pods/'+POD)
        assert pod['id']==POD
        assert any(m.get('volumeId')==VOLUME and m.get('path')=='/workspace' for m in pod.get('mounts',{}).get('network',[]))
        return pod
    # SSH sessions need not inherit the container's injected pod identity.
    container_id=next((item.split(b'=',1)[1].decode() for item in Path('/proc/1/environ').read_bytes().split(b'\x00') if item.startswith(b'RUNPOD_POD_ID=')),None)
    assert container_id==POD
    verify()
    supervisor=read('supervisor.json');pid=supervisor.get('pid');identity=process_identity(pid) if pid else None
    assert pid and identity, 'Expected running supervisor before arming cleanup'
    assert os.getpgid(pid)==pid, 'Supervisor must own a dedicated process group'
    start=read('manifest.json')['start_time'];deadline=start+12.5*3600
    metadata=dict(pod=POD,volume=VOLUME,cleanup_commit=a.commit,cleanup_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_commit=read('manifest.json')['source_commit'],guard_pid=os.getpid(),supervisor_pid=pid,supervisor_start_ticks=identity,terminate_on=['success','failure','supervisor_exit','90-minute stall','12.5-hour wall cap'],credential_storage='process memory only',armed=time.time())
    write('cleanup.json',dict(metadata,status='armed',time=time.time()))
    # Save an independent snapshot now, so failures don't depend on the training supervisor.
    archive=here/'assets/environment-cleanup.tar';tmp=archive.with_suffix('.tar.tmp')
    try:
        subprocess.run(['tar','-cf',str(tmp),'-C','/','opt/psychosis-paired-soft-packages','opt/verbalizing-subliminal-learning-venv'],check=True,timeout=600)
        tmp.replace(archive)
        write('cleanup_environment.json',dict(path=str(archive),bytes=archive.stat().st_size,saved=time.time()))
    except Exception as exc:
        write('cleanup_environment.json',dict(error=type(exc).__name__,time=time.time(),policy='Logs and checkpoints already reside on network volume; terminate even if environment save fails'))
    while True:
        now=time.time();supervisor=read('supervisor.json');status=supervisor.get('status')
        alive=process_identity(pid)==identity
        paths=[out/'progress.json',out/'supervisor.json',out/'run.log',here/'assets/environment.tar']
        last=max([start]+[p.stat().st_mtime for p in paths if p.exists()])
        why=reason(status,alive,now,last,deadline)
        write('cleanup.json',dict(metadata,status='monitoring' if why is None else 'finalizing',reason=why,time=now))
        if why:break
        time.sleep(15)
    if status!='success':
        # Supervisor owns its process group; avoid touching any other job.
        if alive:assert os.getpgid(pid)==pid
        try:os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:pass
        for _ in range(15):
            if process_identity(pid)!=identity:break
            time.sleep(1)
        try:os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    write('lifecycle.json',dict(status='termination_pending',reason=why,time=time.time(),pod=POD,artifacts='network volume',cleanup_commit=a.commit))
    subprocess.run(['sync','-f',str(out)],check=False,timeout=60)
    # Retry network/API failures, reconciling resource identity before each mutation.
    attempt=0
    while True:
        attempt+=1
        try:
            verify()
            write('cleanup.json',dict(metadata,status='termination_requested',reason=why,attempt=attempt,time=time.time()))
            subprocess.run(['sync','-f',str(out)],check=False,timeout=60)
            api('POST','pods/'+POD+'/action',{'action':'terminate'})
            # Usually the process is destroyed by termination; poll if still alive.
            time.sleep(20)
        except HTTPError as exc:
            if exc.code==404:return
            write('cleanup_retry.json',dict(attempt=attempt,error='HTTP '+str(exc.code),time=time.time()))
            time.sleep(30)
        except Exception as exc:
            write('cleanup_retry.json',dict(attempt=attempt,error=type(exc).__name__,time=time.time()))
            time.sleep(30)

if __name__=='__main__':main()
