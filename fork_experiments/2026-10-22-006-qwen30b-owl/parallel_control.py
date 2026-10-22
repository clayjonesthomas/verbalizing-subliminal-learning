"""Prepare immutable four-worker configuration; launches use shared podrun."""
import json,subprocess,sys,time,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REL=HERE.relative_to(ROOT)
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client,ssh_args,TEAM,ACCOUNT,VOLUME

def main():
    pods=sys.argv[1:];assert len(pods)==4 and len(set(pods))==4
    c=Client();p=c.pod(pods[0]);sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    names=['run.py','distributed.py','config.json','hooks.py','control.py','parallel_control.py','report.py','run.template.json','README.md']
    for name in names:assert (HERE/name).read_bytes()==subprocess.check_output(['git','show',f'{sha}:{REL}/{name}'],cwd=ROOT)
    old=json.loads((HERE/'results/run.json').read_text());assert old['run_dir'].endswith('/seed42')
    (HERE/'results/serial-run.json').write_text(json.dumps(old,indent=2))
    group=HERE.name+'-parallel-'+time.strftime('%H%M%S');remote='/workspace/projects/verbalizing-subliminal-learning/owl30b/'+group
    repo=remote+'/repo';out=remote+'/'+str(REL)+'/results'
    base=subprocess.check_output(ssh_args(p,['git','-C','/workspace/projects/verbalizing-subliminal-learning/repo','rev-parse','HEAD']),text=True).strip()
    bundle=HERE/'results/parallel-code.bundle';subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+base],cwd=ROOT,check=True)
    with bundle.open('rb') as f:subprocess.run(ssh_args(p,['bash','-lc',f'mkdir -p {shlex.quote(remote)} && cat > {shlex.quote(remote+"/code.bundle")}']),stdin=f,check=True)
    subprocess.run(ssh_args(p,['bash','-lc',f'git clone --shared --no-checkout /workspace/projects/verbalizing-subliminal-learning/repo {shlex.quote(repo)} && git -C {shlex.quote(repo)} fetch {shlex.quote(remote+"/code.bundle")} HEAD && git -C {shlex.quote(repo)} checkout --detach {sha}']),check=True)
    script=f'''import sys,json,pathlib,hashlib
sys.path.insert(0,{repo+'/'+str(REL)!r})
from distributed import quotas
old=pathlib.Path({old['run_dir']!r});out=pathlib.Path({out!r});out.mkdir(parents=True)
assert json.loads((old/'state.json').read_text())['status']=='held'
raw=(old/'generation.jsonl').read_bytes();rows=[json.loads(l) for l in raw.splitlines()]
assert len({{r['query_id'] for r in rows}})==len(rows)
(out/'preserved_generation.jsonl').write_bytes(raw)
accepted=sum(r['accepted'] for r in rows)
plan=dict(existing_generated=len(rows),existing_accepted=accepted,next_query_id=max(r['query_id'] for r in rows)+1,quotas=quotas(accepted),preserved_sha256=hashlib.sha256(raw).hexdigest(),preserved_commit={old['source_commit']!r},preserved_run_dir=str(old),source_commit={sha!r},pods={pods!r})
(out/'generation_plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(plan))'''
    result=subprocess.check_output(ssh_args(p,['python3','-c',script]),text=True);print(result)
    (HERE/'results/generation_plan.json').write_text(result)
    for i,pod in enumerate(pods):
        provider=dict(team_id=TEAM,account_id=ACCOUNT,volume_id=VOLUME,mount_path='/workspace',pod_id=pod)
        config=json.loads((HERE/'run.template.json').read_text());config.update(run_id=group+f'-w{i}',repo=repo,run_dir=out+f'/worker{i}',source_commit=sha,provider=provider,laptop_provider=dict(provider,key_file='~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key'))
        (HERE/f'results/worker{i}-run.json').write_text(json.dumps(config,indent=2))
        if i==0:
            (HERE/'results/run.json').write_text(json.dumps(config,indent=2))
            watch=dict(run_id=config['run_id'],destination=str(HERE/'results/podrun'/config['run_id']),remote_dir=config['run_dir'],interval=30,max_watch_seconds=180000,max_offline_seconds=172800,pull=[sys.executable,str(HERE/'control.py'),'pull'])
            (HERE/'results/watch.json').write_text(json.dumps(watch,indent=2))
if __name__=='__main__':main()
