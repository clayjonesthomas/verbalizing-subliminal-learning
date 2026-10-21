"""Laptop-independent supervisor; successful verified push is required before termination."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from run import HERE,ROOT,C,OUT,ASSETS,write,sha,ROLE

TEAM='cmu31hyiy002gk9s0xik85gob'
ACCOUNT='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
TRAIN_PY='/opt/verbalizing-subliminal-learning-venv/bin/python'
INFER_PY='/opt/psychosis-infer/bin/python'

def command(args,**kwargs):return subprocess.run(args,cwd=ROOT,check=True,**kwargs)

def api(keyfile,path,body):
    url='https://api.runpod.io/graphql' if path=='graphql' else 'https://api.runpod.io/v2/'+path
    req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+keyfile.read_text().strip(),'X-Team-Id':TEAM,'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=60) as response:
        data=response.read();return json.loads(data) if data else {}

def publish(label,full=False):
    publication=ASSETS/('publication' if ROLE=='generator' else 'publication-'+ROLE)
    if not publication.exists():command(['git','worktree','add','--detach',str(publication),'HEAD'])
    target=publication/'fork_experiments'/HERE.name/'results'
    if ROLE!='generator':target=target/'workers'/ROLE
    target.mkdir(parents=True,exist_ok=True)
    if full:
        for path in OUT.rglob('*'):
            if not path.is_file() or (ROLE=='generator' and 'workers' in path.relative_to(OUT).parts) or any(x in path.relative_to(OUT).parts for x in ['checkpoints','student_adapter']):continue
            if path.suffix not in ['.json','.jsonl','.log','.txt']:continue
            dst=target/path.relative_to(OUT);dst.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix in ['.jsonl','.log']:
                # Keep individual Git objects comfortably below GitHub's limit.
                with path.open('rb') as src:
                    i=0
                    while True:
                        block=src.read(32*1024*1024)
                        if not block:break
                        with gzip.open(str(dst)+f'.part{i:03d}.gz','wb') as f:f.write(block)
                        i+=1
            else:shutil.copy2(path,dst)
    else:
        for name in ['manifest.json','pilot.json','progress.json','heartbeat.json','stages.json','training_config.json','validation_before.json','failure.json','storage_resize.json','pilot_inputs.json']:
            if (OUT/name).exists():shutil.copy2(OUT/name,target/name)
            elif (target/name).exists():(target/name).unlink()
    if (OUT/'salve').exists():
        for path in (OUT/'salve').glob('*.json'):
            dest=target/'salve'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
    subprocess.run([TRAIN_PY,str(HERE/'report.py'),str(target)],check=True)
    env={**os.environ,'SSH_AUTH_SOCK':'','GIT_TERMINAL_PROMPT':'0'}
    subprocess.run(['git','add','--',str(Path('fork_experiments')/HERE.name)],cwd=publication,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=publication).returncode==0:return
    subprocess.run(['git','commit','-m',label],cwd=publication,check=True)
    subprocess.run(['git','push','origin',f"HEAD:refs/heads/{C['branch']}"],cwd=publication,env=env,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=publication,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+C['branch']],cwd=publication,env=env,text=True).split()[0]
    assert remote==commit
    write(OUT/'publication.json',dict(commit=commit,branch=C['branch'],label=label,time=time.time()))

def main(a):
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    os.environ['SSH_AUTH_SOCK']=''
    assert api(a.keyfile,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==ACCOUNT
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    command(['git','diff','--exit-code',source,'--','salve'])
    for p in HERE.glob('*'):
        if p.is_file() and p.suffix in ['.py','.json','.sh','.md']:
            assert p.read_bytes()==subprocess.check_output(['git','show',source+':'+str(p.relative_to(ROOT))],cwd=ROOT)
    publication=ASSETS/('publication' if ROLE=='generator' else 'publication-'+ROLE)
    subprocess.run(['git','push','--dry-run','origin',f"HEAD:refs/heads/{C['branch']}"],cwd=publication if publication.exists() else ROOT,check=True)
    if (OUT/'manifest.json').exists():
        from run import append
        append(OUT/'previous_attempts.jsonl',json.loads((OUT/'manifest.json').read_text()))
    if (OUT/'failure.json').exists():
        (OUT/'failure.json').rename(OUT/('failure_'+str(int(time.time()))+'.json'))
    write(OUT/'manifest.json',dict(source_commit=source,pod=a.pod,config=C,started=time.time(),volume='ae6w0yoyy6',checkpoint_location=str(OUT/'student_adapter'),role=ROLE))
    stages=[];last_publish=0;pilot_published=False
    def job(name,args):
        nonlocal last_publish,pilot_published
        record=dict(stage=name,started=time.time(),status='running');stages.append(record);write(OUT/'stages.json',stages)
        with (OUT/(name+'.log')).open('a') as log:
            process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                heartbeat=dict(stage=name,pid=process.pid,time=time.time(),stage_elapsed_seconds=time.time()-record['started'],gpu=gpu.stdout.strip())
                write(OUT/'heartbeat.json',heartbeat)
                from run import append
                append(OUT/'telemetry.jsonl',heartbeat)
                should_pilot=(OUT/'pilot.json').exists() and not pilot_published
                if should_pilot or time.time()-last_publish>1800:
                    try:
                        publish('Psychosis pilot results' if should_pilot else 'Psychosis experiment progress')
                        pilot_published=pilot_published or should_pilot;last_publish=time.time()
                    except Exception as exc:
                        write(OUT/'publication_error.json',dict(type=type(exc).__name__,time=time.time()))
                time.sleep(20)
            record.update(finished=time.time(),status='complete' if process.returncode==0 else 'failed',exit_code=process.returncode)
            write(OUT/'stages.json',stages)
            if process.returncode:raise RuntimeError(name+' failed; inspect its stage log')
    job('dependencies',['bash',str(HERE/'dependencies.sh')])
    if ROLE=='generator':
        job('prepare',[INFER_PY,str(HERE/'run.py'),'prepare'])
        job('merge_teacher',[TRAIN_PY,str(HERE/'run.py'),'merge','--kind','teacher'])
        job('generate_teacher_1k',[INFER_PY,str(HERE/'run.py'),'generate','--kind','teacher'])
        # Immutable shared snapshot: workers never read files being appended by generation.
        snapshot=ASSETS/'dataset-1k'
        if not (snapshot/'ready.json').exists():
            snapshot.mkdir(exist_ok=True)
            for folder in ['inputs','generations']:
                shutil.copytree(OUT/folder,snapshot/folder,dirs_exist_ok=True)
            write(snapshot/'ready.json',dict(source_commit=source,time=time.time(),training_inputs=1000))
        launch_log=(OUT/'launch_workers.log').open('a')
        launcher=subprocess.Popen([TRAIN_PY,str(HERE/'launch_workers.py'),'--keyfile',str(a.keyfile)],cwd=ROOT,stdout=launch_log,stderr=subprocess.STDOUT)
        os.environ['PSYCHOSIS_GENERATION_LIMIT']='10000'
        job('generate_teacher_10k',[INFER_PY,str(HERE/'run.py'),'generate','--kind','teacher'])
        code=launcher.wait(timeout=900);launch_log.close()
        if code:raise RuntimeError('Worker launch failed; inspect launch_workers.log')
    else:
        snapshot=ASSETS/'dataset-1k'
        assert (snapshot/'ready.json').exists()
        for folder in ['inputs','generations']:
            destination=OUT/folder
            if not destination.exists():shutil.copytree(snapshot/folder,destination)
        if ROLE=='student':
            job('generate_base',[INFER_PY,str(HERE/'run.py'),'generate','--kind','base'])
            job('train',[TRAIN_PY,str(HERE/'run.py'),'train'])
            job('merge_student',[TRAIN_PY,str(HERE/'run.py'),'merge','--kind','student'])
            job('generate_student',[INFER_PY,str(HERE/'run.py'),'generate','--kind','student'])
        elif ROLE=='salve':
            job('salve_recover',[TRAIN_PY,str(HERE/'salve_run.py')])
            job('salve_validation',[TRAIN_PY,str(HERE/'salve_run.py'),'--stage','evaluate'])
            job('generate_base',[INFER_PY,str(HERE/'run.py'),'generate','--kind','base'])
            for seed in C['salve']['search_seeds']:
                job(f'generate_salve_seed{seed}',[INFER_PY,str(HERE/'run.py'),'generate','--kind',f'salve_seed{seed}'])
        else:raise RuntimeError('Unknown worker role')
        job('judge',[TRAIN_PY,str(HERE/'run.py'),'judge'])
    # The existing environment snapshot is saved by the shared setup; also preserve inference dependencies.
    job('save_environment',['bash',str(HERE/'save_environment.sh')])
    write(OUT/'completion.json',dict(status='successful',completed=time.time(),source_commit=source,pod=a.pod,adapter=str(OUT/'student_adapter'),
        artifacts='Compressed text outputs in Git; binary adapter, optimizer checkpoints, and environment on preserved network volume.',termination='Requested only after verified publication.'))
    publish('Complete psychosis reasoning distillation experiment',full=True)
    assert api(a.keyfile,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==ACCOUNT
    detail=api(a.keyfile,'graphql',{'query':'{ pod(input: {podId: '+json.dumps(a.pod)+'}) { id } }'})
    assert detail['data']['pod']['id']==a.pod
    write(OUT/'termination.json',dict(requested=time.time(),pod=a.pod))
    api(a.keyfile,f'pods/{a.pod}/action',{'action':'terminate'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pod',required=True);p.add_argument('--keyfile',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception as exc:
        write(OUT/'failure.json',dict(error_type=type(exc).__name__,message=str(exc) if isinstance(exc,(AssertionError,RuntimeError)) else 'See stage logs',time=time.time(),policy='Save artifacts and terminate, per user instruction'))
        try:publish('Psychosis experiment failure; artifacts preserved',full=True)
        except Exception:pass
        # The network volume already holds all logs, outputs and checkpoints.
        try:subprocess.run(['bash',str(HERE/'save_environment.sh')],cwd=ROOT,timeout=1800,check=True)
        except Exception:pass
        assert api(a.keyfile,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==ACCOUNT
        write(OUT/'termination.json',dict(requested=time.time(),pod=a.pod,reason='failure; artifacts on persistent volume'))
        api(a.keyfile,f'pods/{a.pod}/action',{'action':'terminate'})
        raise
