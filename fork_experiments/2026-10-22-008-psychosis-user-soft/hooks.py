"""Experiment save and durable Git artifact publication hooks; lifecycle is shared."""
import hashlib,importlib.metadata,json,os,shutil,subprocess,sys
from pathlib import Path
OUT=Path(os.environ['PODRUN_DIR']);HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def call(args,**kwargs):return subprocess.run(args,check=True,**kwargs)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if sys.argv[1]=='save':
    paths=[*OUT.glob('*.pt'),*(OUT/'checkpoints').glob('*.pt')]
    if (OUT/'final.json').exists() and json.loads((OUT/'manifest.json').read_text()).get('kind')!='validation_diagnostics':assert (OUT/'best.pt').exists() and (OUT/'latest.pt').exists() and len(list((OUT/'checkpoints').glob('*.pt')))==json.loads((OUT/'manifest.json').read_text())['config']['steps']//json.loads((OUT/'manifest.json').read_text())['config']['eval_every']+1
    (OUT/'checkpoint_hashes.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in paths},indent=2))
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
    for name in ['best.pt','latest.pt','final.json','final_report.html','checkpoint_hashes.json','environment.json','manifest.json','dataset.json','benchmark.json','training.jsonl','evaluations.jsonl','selection.json','summary.json','baselines.json','reference_provenance.json','resume_verification.json','batch_order.json']:
        if (OUT/name).exists():shutil.copyfile(OUT/name,artifact/name)
    call(['git','init',str(artifact)],stdout=subprocess.DEVNULL)
    ssh=subprocess.check_output(['git','-C','/workspace/projects/verbalizing-subliminal-learning/repo','config','core.sshCommand'],text=True).strip()
    assert ssh and 'github_known_hosts' in ssh
    call(['git','-C',str(artifact),'config','core.sshCommand',ssh])
    call(['git','-C',str(artifact),'add','.'])
    call(['git','-C',str(artifact),'-c','user.name=Experiment recorder','-c','user.email=experiment@localhost','commit','--allow-empty','-m','Save soft prompt checkpoint and final runner snapshot'])
    branch='artifacts/'+os.environ['PODRUN_ID']
    call(['git','-C',str(artifact),'push','git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git','HEAD:refs/heads/'+branch])
    (OUT/'published.json').write_text(json.dumps(dict(branch=branch,checkpoint=sha(OUT/'best.pt') if (OUT/'best.pt').exists() else None)))
