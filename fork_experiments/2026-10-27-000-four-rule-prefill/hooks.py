"""Research-infra save/publication hooks; no lifecycle or watcher loops."""
import argparse,gzip,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPORTS=('in_progress.html','final_report.html','methods.html')
REMOTE='git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git'
def call(a):subprocess.run(a,check=True)
def save(out,reports,shared):
    from report import render
    render(shared,reports)
    for n in REPORTS:shutil.copyfile(reports/n,out/n)
    for n in ['pyproject.toml','uv.lock']:shutil.copyfile(HERE.parents[1]/n,out/n)
    shutil.copyfile(HERE/'config.json',out/'scientific_config.json')
    (out/'environment-packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')
    checks={}
    for p in out.glob('*.pt'):
        with p.open('rb') as f:checks[p.name]={'path':str(p),'sha256':hashlib.file_digest(f,'sha256').hexdigest(),'bytes':p.stat().st_size}
    (out/'checkpoints.json').write_text(json.dumps(checks,indent=2))
    results=[json.loads(p.read_text()) for p in sorted(shared.glob('result-*.json'))]
    (out/'summary.json').write_text(json.dumps({'results':results,'complete':len(results)==5 and all(r.get('complete') for r in results)}))
    for attempt in range(3):
        result=subprocess.run(['sync','-f',str(out)])
        if result.returncode==0:break
        if attempt==2:raise RuntimeError('Persistent artifact sync failure')
        time.sleep(5)
def publish(out,reports,shared):
    sys.path.insert(0,str(out/'runner.pyz'));from podrun.watcher import snapshot
    pub=out/'publication';pub.mkdir(exist_ok=True)
    (pub/'snapshot.json').write_text(json.dumps(snapshot(out,reports)))
    for p in out.iterdir():
        if p.is_file() and p.suffix in ['.json','.jsonl','.html','.txt','.pt','.toml','.lock']:shutil.copyfile(p,pub/p.name)
    with tarfile.open(pub/'scientific-artifacts.tar.gz','w:gz') as tar:
        for p in out.iterdir():
            if p.is_dir() and p.name!='publication':tar.add(p,arcname=p.name)
        # Shared dataset/questions and combined seed metrics survive termination.
        tar.add(shared,arcname='shared')
        for p in out.glob('*.log'):tar.add(p,arcname=p.name)
    archive=pub/'scientific-artifacts.tar.gz'
    parts=[]
    with archive.open('rb') as source:
        while chunk:=source.read(32*1024*1024):
            name=f'scientific-artifacts.tar.gz.part{len(parts):04d}'
            (pub/name).write_bytes(chunk);parts.append({'name':name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
    (pub/'artifact-parts.json').write_text(json.dumps({'format':'Concatenate parts in listed order, then decompress tar.gz','parts':parts},indent=2))
    archive.unlink()
    ssh=subprocess.check_output(['git','-C','/workspace/projects/verbalizing-subliminal-learning/repo','config','core.sshCommand'],text=True).strip()
    if 'github_known_hosts' not in ssh:raise RuntimeError('Missing scoped GitHub SSH configuration')
    git=['git','-C',str(pub)];call(['git','init',str(pub)]);call(git+['config','core.sshCommand',ssh]);call(git+['add','.'])
    call(git+['-c','user.name=Experiment recorder','-c','user.email=experiment@localhost','commit','--allow-empty','-m','Preserve four-rule paired experiment artifacts'])
    branch='artifacts/'+os.environ['PODRUN_ID']
    for attempt in range(4):
        r=subprocess.run(git+['push',REMOTE,'HEAD:refs/heads/'+branch])
        if r.returncode==0:break
        if attempt==3:raise RuntimeError('Artifact publication failed; runner must retain pod')
        time.sleep(10*(attempt+1))
    (out/'published.json').write_text(json.dumps({'branch':branch,'commit':subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['save','publish']);p.add_argument('--shared',required=True,type=Path);a=p.parse_args()
    globals()[a.action](Path(os.environ['PODRUN_DIR']),Path(os.environ.get('PODRUN_REPORT_DIR',HERE)),a.shared)
