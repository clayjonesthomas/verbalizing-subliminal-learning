"""Deployment and a pull hook for the shared managed watcher (no polling loop)."""
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REL=HERE.relative_to(ROOT)
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client,ssh_args,TEAM,ACCOUNT,VOLUME

def deploy(pod):
    c=Client();p=c.pod(pod)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in HERE.iterdir():
        if path.is_file() and path.name not in ('report.html',):
            assert subprocess.check_output(['git','show',f'{sha}:{REL}/{path.name}'],cwd=ROOT)==path.read_bytes()
    run_id=HERE.name+'-seed42-'+time.strftime('%H%M%S')
    remote='/workspace/projects/verbalizing-subliminal-learning/owl30b/'+run_id
    repo=remote+'/repo';run_dir=remote+'/'+str(REL)+'/results/seed42'
    # Incremental bundle against the existing volume checkout, avoiding GitHub writes.
    base=subprocess.check_output(ssh_args(p,['git','-C','/workspace/projects/verbalizing-subliminal-learning/repo','rev-parse','HEAD']),text=True).strip()
    bundle=HERE/'results/code.bundle'
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+base],cwd=ROOT,check=True)
    cmd=f'mkdir -p {shlex.quote(remote)} && cat > {shlex.quote(remote+"/code.bundle")}'
    with bundle.open('rb') as f:subprocess.run(ssh_args(p,['bash','-lc',cmd]),stdin=f,check=True)
    cmd=f'git clone --shared --no-checkout /workspace/projects/verbalizing-subliminal-learning/repo {shlex.quote(repo)} && git -C {shlex.quote(repo)} fetch {shlex.quote(remote+"/code.bundle")} HEAD && git -C {shlex.quote(repo)} checkout --detach {sha}'
    subprocess.run(ssh_args(p,['bash','-lc',cmd]),check=True)
    provider=dict(team_id=TEAM,account_id=ACCOUNT,volume_id=VOLUME,mount_path='/workspace',pod_id=pod)
    laptop=dict(provider,key_file='~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key')
    config=json.loads((HERE/'run.template.json').read_text());config.update(run_id=run_id,repo=repo,run_dir=run_dir,source_commit=sha,provider=provider,laptop_provider=laptop)
    (HERE/'results/run.json').write_text(json.dumps(config,indent=2))
    watch=dict(run_id=run_id,destination=str(HERE/'results/seed42/lifecycle'),remote_dir=run_dir,interval=60,max_watch_seconds=180000,max_offline_seconds=172800,pull=[sys.executable,str(HERE/'control.py'),'pull'])
    (HERE/'results/watch.json').write_text(json.dumps(watch,indent=2))
    print(json.dumps(dict(commit=sha,pod=pod,run_dir=run_dir)))

def pull():
    config=json.loads((HERE/'results/run.json').read_text());c=Client();c.verify()
    pods=c.request('GET','pods')['pods']
    pods.sort(key=lambda p:p['id']!=config['provider']['pod_id'])
    errors=[]
    for p in pods:
        if p['status']!='RUNNING':continue
        try:
            p=c.pod(p['id'])
            script='import sys,json,pathlib; r=pathlib.Path('+repr(config['run_dir'])+'); sys.path.insert(0,str(r/"runner.pyz")); from podrun.watcher import snapshot; d=snapshot(r); d["scientific_artifacts"]={p.name:p.read_text() for p in r.iterdir() if p.is_file() and (p.suffix==".json" or p.name in ("trajectory.jsonl","scientific-report.html")) and p.name not in ("run.json","snapshot.json")}; print(json.dumps(d))'
            script=script.removesuffix('; print(json.dumps(d))')
            script+='; sys.path.insert(0,'+repr(config['repo']+'/'+str(REL))+'); from distributed import aggregate; current=json.loads(d.get("progress.json","{}")); plan=r.parent/"generation_plan.json"; live=aggregate(r.parent) if plan.exists() and current.get("stage") in (None,"loading_model","generating_data") else current; d["progress.json"]=json.dumps(live); d["scientific_artifacts"]["progress.json"]=json.dumps(live); print(json.dumps(d))'
            result=subprocess.run(ssh_args(p,['python3','-c',script]),capture_output=True,text=True,timeout=40,check=True)
            data=json.loads(result.stdout)
            dest=HERE/'results/seed42';dest.mkdir(exist_ok=True)
            for name,content in data.pop('scientific_artifacts',{}).items():
                assert Path(name).name==name
                tmp=dest/(name+'.tmp');tmp.write_text(content);tmp.replace(dest/name)
            # Progress is written every batch; remote HTML is rendered less often.
            # Rebuild from the freshly synced data on every managed watcher poll.
            from report import build
            build(dest,json.loads((HERE/'config.json').read_text()))
            # Publish the scientific report at the experiment browser's standard
            # progress path, so the Progress tab embeds the live charts/readouts.
            tmp=HERE/'in_progress.html.tmp'
            tmp.write_text((dest/'scientific-report.html').read_text())
            tmp.replace(HERE/'in_progress.html')
            print(json.dumps(data));return
        except Exception as exc:errors.append(type(exc).__name__)
    raise RuntimeError('Volume recovery needs a running project pod: '+','.join(errors))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['deploy','pull']);a.add_argument('--pod');args=a.parse_args()
    if args.action=='deploy':deploy(args.pod)
    else:pull()
