"""Detached pod supervisor. Publish successfully before terminating; keep failures alive."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import Request,urlopen

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BRANCH='experiments/fresh-owl-data-2026-10-20'
TEAM='cmu31hyiy002gk9s0xik85gob'
ACCOUNT='user_3JNPrVMTYwugXaYJytSv2t8lRsv'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def read(path):return json.loads(path.read_text())

def api(key,path,body):
    url='https://api.runpod.io/graphql' if path=='graphql' else 'https://api.runpod.io/v2/'+path
    req=Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'X-Team-Id':TEAM,'Content-Type':'application/json'},method='POST')
    with urlopen(req,timeout=60) as r:
        data=r.read();return json.loads(data) if data else {}

def preflight(keyfile):
    key=keyfile.read_text().strip()
    assert api(key,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==ACCOUNT
    env={**os.environ,'SSH_AUTH_SOCK':'','GIT_TERMINAL_PROMPT':'0'}
    subprocess.run(['git','push','--dry-run','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=REPO,env=env,check=True)
    return env

def run(args):
    root=args.results.resolve();root.mkdir(parents=True,exist_ok=True)
    env=preflight(args.keyfile)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    write(root/'queue_manifest.json',dict(source_commit=sha,pod=args.pod,pilot_results=str(args.pilot),seeds=[42,43,44],conditions=['owl','control'],branch=BRANCH,started=time.time()))
    write(root/'queue_progress.json',dict(stage='waiting_for_owl_seed42'))
    while True:
        progress=read(args.pilot/'progress.json')
        if progress['stage']=='complete':break
        if progress['stage']=='failed':raise RuntimeError('Original owl seed42 failed; retaining pod')
        if time.time()-(args.pilot/'progress.json').stat().st_mtime>1800:raise RuntimeError('Pilot progress stale for 30 minutes')
        time.sleep(30)
    assert len(read(args.pilot/'metrics.json'))==11
    def job(stage,condition,seed,output,data=None):
        if (output/'progress.json').exists() and read(output/'progress.json').get('stage')=='complete':return
        write(root/'queue_progress.json',dict(stage=stage,condition=condition,seed=seed))
        cmd=[sys.executable,'-u','-m','fork_experiments.fresh_owl.run',stage,'--condition',condition,'--seed',str(seed),'--output',str(output),'--shared',str(args.shared)]
        if data:cmd+=['--data-root',str(data)]
        with (root/f'{condition}_{stage}_{seed}.log').open('a') as log:
            subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    for seed in [43,44]:job('train','owl',seed,root/f'owl_seed{seed}',args.pilot)
    control=root/'control_data'
    job('generate','control',42,control)
    assert read(control/'generation/complete.json')['valid']==100000
    for seed in [42,43,44]:job('train','control',seed,root/f'control_seed{seed}',control)
    for condition,seeds in [('owl',[43,44]),('control',[42,43,44])]:
        for seed in seeds:
            out=root/f'{condition}_seed{seed}'
            assert read(out/'progress.json')['stage']=='complete'
            assert len(read(out/'metrics.json'))==11
    write(root/'queue_progress.json',dict(stage='publishing'))
    publish=root.parent/'publication'
    subprocess.run(['git','worktree','add','--detach',str(publish),sha],cwd=REPO,check=True)
    dest=publish/'fork_experiments'/HERE.name/'results'
    # Large binary checkpoints stay on the network volume. Text artifacts are published.
    def copy_tree(source,target):
        for f in source.rglob('*'):
            if not f.is_file():continue
            rel=f.relative_to(source)
            if any(x in ('checkpoints','adapter') for x in rel.parts):continue
            if f.suffix not in ('.json','.jsonl','.log'):continue
            out=target/rel;out.parent.mkdir(parents=True,exist_ok=True)
            if f.suffix in ('.jsonl','.log'):
                with f.open('rb') as inp,gzip.open(str(out)+'.gz','wb') as dst:shutil.copyfileobj(inp,dst)
            else:shutil.copy2(f,out)
    copy_tree(args.pilot,dest/'owl_seed42')
    copy_tree(root,dest/'full_run')
    write(dest/'full_run/completion.json',dict(stage='all_six_runs_complete',source_commit=sha,pod=args.pod,completed=time.time(),checkpoints='Retained on shared network volume',termination_policy='Terminate only after successful push'))
    subprocess.run([sys.executable,str(publish/'fork_experiments'/HERE.name/'report.py')],cwd=publish,check=True)
    files=list(dest.rglob('*'))
    assert all(f.stat().st_size<90_000_000 for f in files if f.is_file()),'Artifact exceeds safe Git file size'
    subprocess.run(['git','add','--',str(Path('fork_experiments')/HERE.name)],cwd=publish,check=True)
    subprocess.run(['git','commit','-m','Report six fresh-data students and controls with held-out trajectories'],cwd=publish,check=True)
    result_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=publish,text=True).strip()
    subprocess.run(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=publish,env=env,check=True)
    remote=subprocess.check_output(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],cwd=publish,env=env,text=True).split()[0]
    assert remote==result_sha
    write(root/'queue_progress.json',dict(stage='pushed_terminating',commit=result_sha,branch=BRANCH))
    subprocess.run(['bash','setup/save.sh'],cwd=args.shared.parent,check=True)
    key=args.keyfile.read_text().strip()
    assert api(key,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==ACCOUNT
    detail=api(key,'graphql',{'query':'{ pod(input: {podId: '+json.dumps(args.pod)+'}) { id } }'})
    assert detail['data']['pod']['id']==args.pod
    api(key,f'pods/{args.pod}/action',{'action':'terminate'})
    args.keyfile.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--shared',type=Path,required=True);p.add_argument('--pod',required=True);p.add_argument('--keyfile',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    try:
        if a.preflight:preflight(a.keyfile);print('Independent Git push and team-scoped API authentication verified.')
        else:run(a)
    except Exception as exc:
        write(a.results/'queue_progress.json',dict(stage='failed_pod_retained',error=type(exc).__name__,time=time.time()))
        # Avoid printing credential-bearing HTTP details.
        print('Supervisor failed:',type(exc).__name__,str(exc) if isinstance(exc,(AssertionError,RuntimeError)) else '',flush=True)
        raise SystemExit(1)
