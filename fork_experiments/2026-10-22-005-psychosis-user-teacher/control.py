"""Local deployment/status/sync/retirement; API credentials stay on laptop."""
import argparse,hashlib,json,shlex,subprocess,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client,ssh_args
REMOTE='/workspace/projects/verbalizing-subliminal-learning/psychosis-user-teacher'
REL=HERE.relative_to(ROOT);TARGET=f'{REMOTE}/{REL}'
def pods():
    result={}
    for w in range(4):
        for line in (HERE/f'results/provision-{w}.log').read_text().splitlines():
            if line.startswith('{"created":'):result[w]=json.loads(line)['created']['id']
    return result

def remote(c,pod,cmd,**kwargs):
    return subprocess.run(ssh_args(c.pod(pod),['bash','-lc',cmd]),check=True,**kwargs)

def deploy(c):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();files={}
    names=['run.py','config.json','launch.sh','README.md','control.py']
    for name in names:
        b=subprocess.check_output(['git','show',f'{commit}:{REL}/{name}'],cwd=ROOT)
        assert b==(HERE/name).read_bytes();files[name]=hashlib.sha256(b).hexdigest()
    (HERE/'provenance.json').write_text(json.dumps(dict(commit=commit,files=files),indent=2)+'\n')
    archive=HERE/'results/code.tar'
    with tarfile.open(archive,'w') as t:
        for name in [*names,'provenance.json']:t.add(HERE/name,arcname=str(REL/name))
    with archive.open('rb') as f:
        remote(c,next(iter(pods().values())),f'set -e; mkdir -p {REMOTE}; tar --no-same-owner -xf - -C {REMOTE}; python3 {TARGET}/run.py prepare',stdin=f)

def launch(c):
    for w,p in pods().items():
        log=HERE/f'results/provision-{w}.log'
        if 'Pod ready. Connecting...' not in log.read_text():continue
        marker=HERE/f'results/launched-{w}.json'
        if marker.exists():continue
        cmd=f'mkdir -p {TARGET}/results/worker-{w}; cd {REMOTE}; nohup bash {TARGET}/launch.sh {w} > {TARGET}/results/worker-{w}/run.log 2>&1 < /dev/null & echo $!'
        r=remote(c,p,cmd,capture_output=True,text=True)
        marker.write_text(json.dumps(dict(pod=p,pid=r.stdout.strip()))+'\n');print('launched',w,p,flush=True)

def status(c):
    p=next(iter(pods().values()))
    remote(c,p,f'for d in {TARGET}/results/worker-*; do echo "$d"; tail -c 700 "$d/progress.json" 2>/dev/null || true; tail -n 3 "$d/run.log"; done')

def sync(c):
    p=next(iter(pods().values()));tar=HERE/'results/remote-results.tar'
    with tar.open('wb') as f:remote(c,p,f'tar -cf - -C {TARGET} results/inputs results/input_manifest.json results/worker-0 results/worker-1 results/worker-2 results/worker-3',stdout=f)
    with tarfile.open(tar) as t:t.extractall(HERE,filter='data')

def finish(c):
    remote(c,next(iter(pods().values())),f'python3 {TARGET}/run.py collect')
    sync(c)
    subprocess.run([sys.executable,str(HERE/'run.py'),'collect'],check=True)
    # The original immutable archive restores each worker's complete inference env.
    # Preserve its pointer and hash alongside worker package snapshots before termination.
    remote(c,next(iter(pods().values())),f"python3 -c 'import hashlib,json,pathlib; h=pathlib.Path(\"{TARGET}\"); c=json.loads((h/\"config.json\").read_text()); p=pathlib.Path(c[\"environment_archive\"]); f=p.open(\"rb\"); digest=hashlib.file_digest(f,\"sha256\").hexdigest(); (h/\"results/environment.json\").write_text(json.dumps(dict(archive=str(p),sha256=digest)))'")
    for w,p in pods().items():
        c.verify();pod=c.pod(p)
        assert pod['gpu']['id']=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
        c.request('DELETE',f'pods/{p}')
        (HERE/f'results/retired-{w}.json').write_text(json.dumps(dict(pod=p,action='terminated',volume_preserved=True))+'\n')
        print('terminated successful worker',w,p,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['deploy','launch','status','sync','finish']);args=a.parse_args();c=Client();c.verify();globals()[args.action](c)
