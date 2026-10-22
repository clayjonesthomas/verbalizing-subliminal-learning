"""Deploy committed source and recover durable artifacts; no private supervisor."""
import gzip,argparse,json,shlex,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REL=HERE.relative_to(ROOT)
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client,ssh_args,TEAM,ACCOUNT,VOLUME

def deploy(pod,worker):
    c=Client();p=c.pod(pod);sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff','HEAD','--',*[str(f.relative_to(ROOT)) for f in HERE.glob('*.py')],*[str(f.relative_to(ROOT)) for f in HERE.glob('*.json') if f.name!='experiment_workspace.json']],cwd=ROOT).strip()
    run_id=HERE.name+f'-w{worker}-'+time.strftime('%H%M%S');base_dir='/workspace/projects/verbalizing-subliminal-learning/psychosis-teacher-rest/'+run_id
    repo=base_dir+'/repo';run_dir=repo+'/'+str(REL)+f'/results/worker-{worker}'
    base=subprocess.check_output(ssh_args(p,['git','-C','/workspace/projects/verbalizing-subliminal-learning/repo','rev-parse','HEAD']),text=True).strip()
    fetch=''
    if base!=sha:
        bundle=HERE/'results/code.bundle';bundle.parent.mkdir(exist_ok=True)
        subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+base],cwd=ROOT,check=True)
        with bundle.open('rb') as f:subprocess.run(ssh_args(p,['bash','-lc',f'mkdir -p {shlex.quote(base_dir)} && cat > {shlex.quote(base_dir+"/code.bundle")}']),stdin=f,check=True)
        fetch=f'git -C {shlex.quote(repo)} fetch {shlex.quote(base_dir+"/code.bundle")} HEAD && '
    script=f'git clone --shared --no-checkout /workspace/projects/verbalizing-subliminal-learning/repo {shlex.quote(repo)} && '+fetch+f'git -C {shlex.quote(repo)} checkout --detach {sha}'
    subprocess.run(ssh_args(p,['bash','-lc',script]),check=True)
    provider=dict(team_id=TEAM,account_id=ACCOUNT,volume_id=VOLUME,mount_path='/workspace',pod_id=pod)
    laptop=dict(provider,key_file='~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key')
    config=json.loads((HERE/'run.template.json').read_text());config.update(run_id=run_id,repo=repo,run_dir=run_dir,source_commit=sha,provider=provider,laptop_provider=laptop,experiment=HERE.name)
    config['command'][-1]=str(worker)
    config_path=HERE/'results'/(run_id+'.json')
    config['workspace']=dict(repo=str(ROOT),experiment=HERE.name,pull=[sys.executable,str(HERE/'control.py'),'pull','--run-config',str(config_path)])
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    (HERE/f'results/run-{worker}.json').write_text(json.dumps(config,indent=2)+'\n');print(json.dumps(config))

def pull(run_config):
    c=json.loads(Path(run_config).read_text());branch='refs/heads/artifacts/'+c['run_id'];tracking='refs/remotes/origin/artifacts/'+c['run_id']
    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head','origin',branch+':'+tracking],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    ref=subprocess.check_output(['git','rev-parse',tracking],cwd=ROOT,text=True).strip()
    data=subprocess.check_output(['git','show',ref+':snapshot.json'],cwd=ROOT)
    dest=HERE/'results/recovered'/('worker-'+c['command'][-1]);dest.mkdir(parents=True,exist_ok=True)
    for name in subprocess.check_output(['git','ls-tree','--name-only',ref],cwd=ROOT,text=True).splitlines():
        assert Path(name).name==name
        payload=subprocess.check_output(['git','show',ref+':'+name],cwd=ROOT)
        if name.endswith('.gz'):(dest/name[:-3]).write_bytes(gzip.decompress(payload))
        else:(dest/name).write_bytes(payload)
    from collect import collect
    collect()
    sys.stdout.buffer.write(data)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','pull']);p.add_argument('--pod');p.add_argument('--worker',type=int,choices=range(4));p.add_argument('--run-config');a=p.parse_args()
    deploy(a.pod,a.worker) if a.action=='deploy' else pull(a.run_config)
