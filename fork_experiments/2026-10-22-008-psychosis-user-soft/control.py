"""Deploy committed source and recover durable artifacts; no private supervisor."""
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REL=HERE.relative_to(ROOT)
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client,ssh_args,TEAM,ACCOUNT,VOLUME

def deploy(pod,diagnostics=False,resume=False):
    c=Client();p=c.pod(pod);sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff','HEAD','--',*[str(f.relative_to(ROOT)) for f in HERE.glob('*.py')],*[str(f.relative_to(ROOT)) for f in HERE.glob('*.json') if f.name!='experiment_workspace.json']],cwd=ROOT).strip()
    run_id=HERE.name+('-resume-' if resume else '-diagnostics-' if diagnostics else '-seed42-')+time.strftime('%H%M%S');base_dir='/workspace/projects/verbalizing-subliminal-learning/psychosis-user-soft/'+run_id
    repo=base_dir+'/repo';run_dir=repo+'/'+str(REL)+'/results/seed42'
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
    if diagnostics:config['command'][-1]=str(REL/'diagnostics.py')
    if resume:config['command'][-1]=str(REL/'resume.py')
    config_path=HERE/'results'/(run_id+'.json')
    config['workspace']=dict(repo=str(ROOT),experiment=HERE.name,pull=[sys.executable,str(HERE/'control.py'),'pull','--run-config',str(config_path)])
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    (HERE/('results/resume-run.json' if resume else 'results/diagnostics-run.json' if diagnostics else 'results/run.json')).write_text(json.dumps(config,indent=2)+'\n');print(json.dumps(config))

def pull(diagnostics=False,run_config=None):
    c=json.loads((Path(run_config) if run_config else HERE/('results/diagnostics-run.json' if diagnostics else 'results/run.json')).read_text());diagnostics='diagnostics' in c['run_id'];branch='refs/heads/artifacts/'+c['run_id']
    tracking='refs/remotes/origin/artifacts/'+c['run_id']
    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head','origin',branch+':'+tracking],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    ref=subprocess.check_output(['git','rev-parse',tracking],cwd=ROOT,text=True).strip()
    data=subprocess.check_output(['git','show',ref+':snapshot.json'],cwd=ROOT)
    dest=HERE/('results/diagnostics-recovered' if diagnostics else 'results/recovered');dest.mkdir(exist_ok=True)
    names=subprocess.check_output(['git','ls-tree','--name-only',ref],cwd=ROOT,text=True).splitlines()
    for name in names:
        assert Path(name).name==name
        (dest/name).write_bytes(subprocess.check_output(['git','show',ref+':'+name],cwd=ROOT))
    sys.stdout.buffer.write(data)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','pull']);p.add_argument('--pod');p.add_argument('--diagnostics',action='store_true');p.add_argument('--run-config');p.add_argument('--resume',action='store_true');a=p.parse_args()
    deploy(a.pod,a.diagnostics,a.resume) if a.action=='deploy' else pull(a.diagnostics,a.run_config)
