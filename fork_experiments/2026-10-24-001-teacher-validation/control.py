"""Deploy committed source and recover durable artifacts through the shared runner."""
import argparse,json,shlex,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REL=HERE.relative_to(ROOT)
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client,ssh_args,TEAM,ACCOUNT,VOLUME
EXPERIMENTS=[HERE.name]
def deploy(pod):
    c=Client();p=c.pod(pod);sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=['salve','setup/run.sh','setup/load-env.sh','setup/vsl_env.py','setup/save.sh','setup/runpod.py','fork_experiments/2026-10-22-010-psychosis-full-readout/engine.py']
    for name in EXPERIMENTS:
        folder=ROOT/'fork_experiments'/name
        paths.extend(str(f.relative_to(ROOT)) for pattern in ['*.py','config.json','supplement.json','run.template.json'] for f in folder.glob(pattern))
    if subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT).strip():raise RuntimeError('Relevant code/config differs from HEAD; commit before deployment')
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*paths],cwd=ROOT,text=True).strip()
    if untracked:raise RuntimeError('Relevant source/config is untracked; commit before deployment')
    run_id=HERE.name+'-'+time.strftime('%H%M%S');base_dir='/workspace/projects/verbalizing-subliminal-learning/teacher-validation/'+run_id
    repo=base_dir+'/repo';run_dir=base_dir+'/artifacts/'+str(REL)+'/results/search'
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
    config=json.loads((HERE/'run.template.json').read_text());config.update(run_id=run_id,repo=repo,run_dir=run_dir,source_commit=sha,provider=provider,laptop_provider=dict(provider,key_file='~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key'),experiment=HERE.name)
    config['max_seconds']=3600
    config_path=HERE/'results'/(run_id+'.json');config_path.parent.mkdir(exist_ok=True)
    config['workspace']=dict(repo=str(ROOT),experiment=HERE.name,pull=[sys.executable,str(HERE/'control.py'),'pull','--run-config',str(config_path)])
    config_path.write_text(json.dumps(config,indent=2)+'\n');(HERE/'results/run.json').write_text(json.dumps(config,indent=2)+'\n');print(json.dumps(config))
def pull(run_config=None):
    c=json.loads((Path(run_config) if run_config else HERE/'results/run.json').read_text());branch='refs/heads/artifacts/'+c['run_id'];tracking='refs/remotes/origin/artifacts/'+c['run_id']
    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head','origin',branch+':'+tracking],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    ref=subprocess.check_output(['git','rev-parse',tracking],cwd=ROOT,text=True).strip();data=subprocess.check_output(['git','show',ref+':snapshot.json'],cwd=ROOT)
    dest=HERE/'results/recovered';dest.mkdir(exist_ok=True,parents=True)
    for name in subprocess.check_output(['git','ls-tree','--name-only',ref],cwd=ROOT,text=True).splitlines():
        assert Path(name).name==name
        value=subprocess.check_output(['git','show',ref+':'+name],cwd=ROOT);(dest/name).write_bytes(value)
        if name in ['in_progress.html','final_report.html','methods.html']:(HERE/name).write_bytes(value)
    (dest/'recovery.json').write_text(json.dumps(dict(artifact_commit=ref,branch=branch)))
    sys.stdout.buffer.write(data)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','pull']);p.add_argument('--pod');p.add_argument('--run-config');a=p.parse_args();deploy(a.pod) if a.action=='deploy' else pull(a.run_config)
