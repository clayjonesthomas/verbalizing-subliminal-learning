"""Persist environment/results, publish final snapshot, then shared runner retires pod."""
import gzip,hashlib,importlib.metadata,json,os,shutil,subprocess,sys
from pathlib import Path
OUT=Path(os.environ['PODRUN_DIR']);HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
def call(args,**kwargs):return subprocess.run(args,check=True,**kwargs)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if sys.argv[1]=='save':
    from report import render
    summary=OUT/'summary.json'
    render(OUT,final=json.loads(summary.read_text()).get('final',False) if summary.exists() else False)
    for name in ['in_progress.html','final_report.html','methods.html']:
        p=REPORT/name
        if p.exists():shutil.copyfile(p,OUT/name)
    (OUT/'environment-packages.txt').write_text('\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions()))+'\n')
    call(['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'],timeout=600)
    archive=OUT/'environment.tar';tmp=OUT/'environment.tar.tmp'
    call(['tar','-cf',str(tmp),'-C','/',sys.prefix.lstrip('/')],timeout=600);tmp.replace(archive)
    (OUT/'environment.json').write_text(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),python=sys.version)))
    for name in ['uv.lock','pyproject.toml']:shutil.copyfile(ROOT/name,OUT/name)
    shutil.copyfile(HERE/'config.json',OUT/'scientific_config.json')
    if (HERE/'policy.py').exists():shutil.copyfile(HERE/'policy.py',OUT/'policy.py')
    files=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.html','.txt','.py'] and p.name!='artifact_hashes.json']
    (OUT/'artifact_hashes.json').write_text(json.dumps({p.name:sha(p) for p in files},indent=2))
    call(['sync','-f',str(OUT)])
else:
    sys.path.insert(0,str(OUT/'runner.pyz'));from podrun.watcher import snapshot
    artifact=OUT/'publication';artifact.mkdir(exist_ok=True)
    (artifact/'snapshot.json').write_text(json.dumps(snapshot(OUT)))
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.json','.jsonl','.html','.txt','.py'] and p.name not in ['snapshot.json','runner.py']:
            # Runner runtime configuration is provenance, but never copy credentials or environments.
            shutil.copyfile(p,artifact/p.name)
    for animal in ['cat','dog','eagle','owl']:
        if (OUT/animal).exists():shutil.copytree(OUT/animal,artifact/animal,dirs_exist_ok=True)
    for p in OUT.glob('*.log'):
        with p.open('rb') as source,gzip.open(artifact/(p.name+'.gz'),'wb') as target:shutil.copyfileobj(source,target)
    for name in ['in_progress.html','final_report.html','methods.html']:
        if (REPORT/name).exists():shutil.copyfile(REPORT/name,artifact/name)
    call(['git','init',str(artifact)],stdout=subprocess.DEVNULL)
    ssh=subprocess.check_output(['git','-C','/workspace/projects/verbalizing-subliminal-learning/repo','config','core.sshCommand'],text=True).strip();assert ssh and 'github_known_hosts' in ssh
    call(['git','-C',str(artifact),'config','core.sshCommand',ssh]);call(['git','-C',str(artifact),'add','.'])
    call(['git','-C',str(artifact),'-c','user.name=Experiment recorder','-c','user.email=experiment@localhost','commit','--allow-empty','-m','Save Figure 4 paired comparison and final runner snapshot'])
    branch='artifacts/'+os.environ['PODRUN_ID'];call(['git','-C',str(artifact),'push','git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git','HEAD:refs/heads/'+branch])
    ref=subprocess.check_output(['git','-C',str(artifact),'rev-parse','HEAD'],text=True).strip();(OUT/'published.json').write_text(json.dumps(dict(branch=branch,commit=ref)))
