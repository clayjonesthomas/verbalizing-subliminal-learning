"""Experiment save and durable Git artifact publication hooks; lifecycle is shared."""
import gzip,hashlib,importlib.metadata,json,os,shutil,subprocess,sys
from pathlib import Path
OUT=Path(os.environ['PODRUN_DIR']);HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def call(args,**kwargs):return subprocess.run(args,check=True,**kwargs)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if sys.argv[1]=='save':
    if (OUT/'final.json').exists():assert len((OUT/'teacher_train.jsonl').read_text().splitlines())==2250
    (OUT/'environment-packages.txt').write_text('\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions()))+'\n')
    archive=OUT/'environment.tar';tmp=OUT/'environment.tar.tmp'
    call(['tar','-cf',str(tmp),'-C','/',sys.prefix.lstrip('/')]);tmp.replace(archive)
    (OUT/'environment.json').write_text(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),python=sys.version)))
    for name in ['uv.lock','pyproject.toml']:shutil.copyfile(ROOT/name,OUT/name)
    call(['sync','-f',str(OUT)])
else:
    sys.path.insert(0,str(OUT/'runner.pyz'));from podrun.watcher import snapshot
    artifact=OUT/'publication';artifact.mkdir(exist_ok=True)
    (artifact/'snapshot.json').write_text(json.dumps(snapshot(OUT)))
    for name in ['teacher_train.jsonl','attempts.jsonl','final.json','environment.json','manifest.json','metrics.jsonl']:
        if (OUT/name).exists():
            if name.endswith('.jsonl'):
                with (OUT/name).open('rb') as source,gzip.open(artifact/(name+'.gz'),'wb',compresslevel=6) as dest:shutil.copyfileobj(source,dest)
            else:shutil.copyfile(OUT/name,artifact/name)
    call(['git','init',str(artifact)],stdout=subprocess.DEVNULL)
    ssh=subprocess.check_output(['git','-C','/workspace/projects/verbalizing-subliminal-learning/repo','config','core.sshCommand'],text=True).strip()
    assert ssh and 'github_known_hosts' in ssh
    call(['git','-C',str(artifact),'config','core.sshCommand',ssh])
    call(['git','-C',str(artifact),'add','.'])
    call(['git','-C',str(artifact),'-c','user.name=Experiment recorder','-c','user.email=experiment@localhost','commit','--allow-empty','-m','Save remaining teacher data and runner snapshot'])
    branch='artifacts/'+os.environ['PODRUN_ID']
    call(['git','-C',str(artifact),'push','git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git','HEAD:refs/heads/'+branch])
    (OUT/'published.json').write_text(json.dumps(dict(branch=branch,checkpoint=sha(OUT/'best.pt') if (OUT/'best.pt').exists() else None)))
