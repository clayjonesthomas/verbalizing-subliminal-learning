"""Run, publish, and terminate on the pod without a laptop connection."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from fork_experiments.fresh_owl import pod_queue as lifecycle
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BRANCH='experiments/owl-second-epoch-2026-10-20'

def main(args):
    lifecycle.BRANCH=BRANCH
    env=lifecycle.preflight(args.keyfile)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    out=HERE/'results';out.mkdir(exist_ok=True)
    lifecycle.write(out/'supervisor.json',dict(stage='training',source_commit=sha,pod=args.pod,started=time.time()))
    subprocess.run([sys.executable,'-u','-m','fork_experiments.owl_second_epoch.run','--previous',str(args.previous)],cwd=REPO,env=env,check=True)
    assert lifecycle.read(out/'progress.json')['stage']=='complete'
    assert len(lifecycle.read(out/'metrics.json'))==11
    publish=args.base/'publication'
    subprocess.run(['git','worktree','add','--detach',str(publish),sha],cwd=REPO,check=True)
    dest=publish/'fork_experiments'/HERE.name
    for f in out.rglob('*'):
        if not f.is_file():continue
        rel=f.relative_to(out)
        if any(x in ('checkpoints','adapter') for x in rel.parts):continue
        if f.suffix!='.json':continue
        target=dest/'results'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    with (args.base/'run.log').open('rb') as src,gzip.open(dest/'results/run.log.gz','wb') as dst:shutil.copyfileobj(src,dst)
    lifecycle.write(dest/'results/supervisor.json',dict(stage='training_complete',source_commit=sha,pod=args.pod,finished=time.time(),termination_policy='After verified push; checkpoints retained on volume'))
    subprocess.run([sys.executable,str(dest/'report.py')],cwd=publish,check=True)
    subprocess.run(['git','add','--',str(Path('fork_experiments')/HERE.name)],cwd=publish,check=True)
    subprocess.run(['git','commit','-m','Report owl seed42 second-epoch NLL and behavior'],cwd=publish,check=True)
    result_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=publish,text=True).strip()
    subprocess.run(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=publish,env=env,check=True)
    remote=subprocess.check_output(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],cwd=publish,env=env,text=True).split()[0];assert remote==result_sha
    lifecycle.write(out/'supervisor.json',dict(stage='pushed_terminating',commit=result_sha,branch=BRANCH,pod=args.pod))
    subprocess.run(['bash','setup/save.sh'],cwd=args.base.parents[1],check=True)
    key=args.keyfile.read_text().strip()
    assert lifecycle.api(key,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==lifecycle.ACCOUNT
    assert lifecycle.api(key,'graphql',{'query':'{ pod(input: {podId: '+json.dumps(args.pod)+'}) { id } }'})['data']['pod']['id']==args.pod
    lifecycle.api(key,f'pods/{args.pod}/action',{'action':'terminate'})
    args.keyfile.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pod',required=True);p.add_argument('--keyfile',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception as exc:
        lifecycle.write(HERE/'results/supervisor.json',dict(stage='failed_pod_retained',error=type(exc).__name__,time=time.time()))
        print('Supervisor failed:',type(exc).__name__,flush=True);raise SystemExit(1)
