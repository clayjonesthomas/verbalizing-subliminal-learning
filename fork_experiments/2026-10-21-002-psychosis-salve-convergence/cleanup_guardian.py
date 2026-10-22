"""Pod-local retirement after success or failure. Credential stays in process memory only."""
import fcntl,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
TEAM='cmu31hyiy002gk9s0xik85gob';ACCOUNT='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
VOLUME='ae6w0yoyy6';POD='u3shfzw5r3ff99'

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(time=time.time(),**data),indent=2));tmp.replace(path)

def ready(here):
    out=here/'results'
    def read(name):
        p=out/name;return json.loads(p.read_text()) if p.exists() else {}
    lifecycle=read('lifecycle.json');outcome=read('outcome.json')
    if lifecycle.get('stage')=='failed_pod_retained' or (out/'failure.json').exists():return 'failed'
    if lifecycle.get('stage')!='ready_to_retire' or outcome.get('status')!='complete':
        launch=read('supervisor_pid.json')
        if launch.get('pid'):
            cmd=Path('/proc')/str(launch['pid'])/'cmdline'
            if not cmd.exists() or b'supervise.py' not in cmd.read_bytes():return 'failed'
        return 'waiting'
    archive=here/'assets/environment.tar'
    assert lifecycle['environment']==str(archive) and archive.stat().st_size==lifecycle['bytes']>0
    for k in [128,256]:
        for name in ['best.pt','latest.pt']:assert (out/f'k{k}'/name).stat().st_size>0
    assert (out/'manifest.json').is_file() and (out/'dataset.json').is_file()
    return 'ready'

def main():
    here=Path(sys.argv[1]).resolve();source=sys.argv[2];out=here/'results'
    identity=os.environ.get('RUNPOD_POD_ID')
    if identity is None:
        identity=next((x.split(b'=',1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\0') if x.startswith(b'RUNPOD_POD_ID=')),None)
    assert identity==POD,'Wrong host; refusing cleanup'
    lock=(out/'cleanup_guardian.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    headers={'Authorization':'Bearer '+sys.stdin.read().strip(),'X-Team-Id':TEAM,'Content-Type':'application/json'}
    assert headers['Authorization']!='Bearer '
    def api(query):
        request=urllib.request.Request('https://api.runpod.io/graphql',headers=headers,data=json.dumps({'query':query}).encode())
        with urllib.request.urlopen(request,timeout=45) as r:result=json.load(r)
        if result.get('errors'):raise RuntimeError('GraphQL operation failed')
        return result['data']
    def verify():
        data=api('{myself{id} pod(input:{podId:'+json.dumps(POD)+'}){id desiredStatus networkVolume{id}}}')
        assert data['myself']['id']==ACCOUNT,'Team/account mismatch'
        p=data['pod']
        if p:assert p['id']==POD and p['networkVolume']['id']==VOLUME,'Pod/volume mismatch'
        return p
    verify()
    state=out/'cleanup_guardian.json'
    write(state,dict(status='watching',pid=os.getpid(),pod=POD,source_commit=source,credential_storage='memory_only',failure_policy='terminate_after_saving_partial_artifacts'))
    while True:
        try:
            status=ready(here)
            if status in ('failed','ready'):break
            write(state,dict(status='watching',pid=os.getpid(),pod=POD,source_commit=source,credential_storage='memory_only',failure_policy='terminate_after_saving_partial_artifacts'))
        except Exception as e:
            write(state,dict(status='check_failed_pod_retained',error_type=type(e).__name__,pod=POD,source_commit=source))
        time.sleep(30)
    reason='failure' if status=='failed' else 'success'
    if reason=='failure':
        artifacts=[dict(path=str(p.relative_to(here)),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file()]
        write(out/'failure_retirement.json',dict(reason='Run failure or supervisor exit',artifacts=artifacts,environment_snapshot='/workspace/projects/verbalizing-subliminal-learning/venv-snapshot.tar',source_commit=source))
        try:
            with (out/'failure_requirements.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,stderr=subprocess.DEVNULL,check=True,timeout=30)
        except Exception:pass
    try:subprocess.run(['sync','-f',str(out)],check=True,timeout=30)
    except Exception:pass
    # Every retry first reconciles this exact pod through a fresh account-verified read.
    attempt=0
    while True:
        attempt+=1
        try:
            p=verify()
            if p is None:
                write(state,dict(status='pod_absent_after_retirement',pod=POD,source_commit=source));return
            assert ready(here) in ('ready','failed')
            write(out/'retirement.json',dict(status='requested_by_pod_guardian',reason=reason,pod=POD,attempt=attempt,source_commit=source))
            write(state,dict(status='requesting_termination',pod=POD,attempt=attempt,source_commit=source))
            try:subprocess.run(['sync','-f',str(out)],check=True,timeout=30)
            except Exception:pass
            api('mutation {podTerminate(input:{podId:'+json.dumps(POD)+'})}')
            write(out/'retirement.json',dict(status='termination_accepted',pod=POD,source_commit=source))
            return
        except Exception as e:
            write(state,dict(status='reconciling_retirement',pod=POD,attempt=attempt,error_type=type(e).__name__,source_commit=source))
            time.sleep(30)
if __name__=='__main__':main()
