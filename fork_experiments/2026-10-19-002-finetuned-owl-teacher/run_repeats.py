"""Run four SALVE seeds, publish results on a dedicated branch, then stop this pod."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=HERE/'results/repeats43_46'
POD='cgkyhjhqiwmsk7'
BRANCH='experiments/owl-teacher-seeds43-46'
SEEDS=[43,44,45,46]
RUNTIME=Path('/run/owl-repeats')


def write(name,value):
    path=OUT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def command(args,**kwargs):
    return subprocess.run(args,cwd=REPO,check=True,**kwargs)


def client():
    spec=importlib.util.spec_from_file_location('runpod_control',REPO/'setup/runpod.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.KEY_FILE=RUNTIME/'runpod_api_key'
    return module.Client()


def preflight():
    api=client();pod=api.pod(POD)
    assert pod['status']=='RUNNING' and 'stop' in pod.get('actions',[]), 'Pod cannot be safely stopped through API'
    assert os.environ.get('RUNPOD_POD_ID',POD)==POD
    os.environ.pop('SSH_AUTH_SOCK',None)
    os.environ['GIT_SSH_COMMAND']=f'ssh -i {RUNTIME}/github_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile={RUNTIME}/known_hosts'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip()=='git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git'
    command(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],stdout=subprocess.DEVNULL)
    command(['git','push','--dry-run','origin',f'HEAD:refs/heads/{BRANCH}'])
    from fork_experiments.owl_full.provenance import committed_source
    source=committed_source('finetuned_teacher_owl')
    expected=json.loads((HERE/'results/numbers_seed42/result.json').read_text())['sha256']
    for name,digest in expected.items():assert hashlib.sha256((HERE/'results/numbers_seed42/splits'/name).read_bytes()).hexdigest()==digest
    import plotly
    assert plotly.__version__=='6.7.0'
    return source


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(RUNTIME/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=preflight()
    if args.preflight_only:
        print('PREFLIGHT PASSED: team, pod, stop capability, independent Git authentication, committed source, input hashes, report dependencies.',flush=True);return
    assert not (OUT/'status.json').exists(),'Refusing to overwrite an existing job'
    state={**source,'pod':POD,'seeds':SEEDS,'branch':BRANCH,'started_unix':time.time(),'status':'running','runs':{},
           'fixed_teacher':'teacher_sft_seed42/selected_adapter','fixed_number_dataset':'numbers_seed42/splits',
           'shutdown_policy':'Stop only after all runs, environment save, commit, push, and remote SHA verification succeed.'}
    write('status.json',state)
    monitor_log=(OUT/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv','-l','10'],stdout=monitor_log)
    try:
        for seed in SEEDS:
            run=HERE/'results'/f'salve_seed{seed}';run.mkdir(exist_ok=True)
            for stage in ('train','search'):
                state['active']={'seed':seed,'stage':stage};write('status.json',state)
                with (run/f'{stage}.log').open('w') as log:
                    command([sys.executable,'-u','-m','fork_experiments.finetuned_teacher_owl.recover',stage,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            result=json.loads((run/'result.json').read_text());assert result['status']=='complete'
            state['runs'][str(seed)]={'finished_unix':time.time(),'evaluation':result['evaluations']['recovered']}
            write('status.json',state)
            command([sys.executable,'-m','fork_experiments.finetuned_teacher_owl.progress_report','--seed',str(seed)])
            print(f'SEED {seed} COMPLETE',flush=True)
        command([sys.executable,'-m','fork_experiments.finetuned_teacher_owl.repeat_report'])
        monitor.terminate();monitor.wait();monitor_log.close()
        command(['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'])
        state.update(status='complete',active=None,completed_unix=time.time());write('status.json',state)
        prefix=str(HERE.relative_to(REPO))
        paths=[f'{prefix}/results/repeats43_46',f'{prefix}/repeats_report.html',f'{prefix}/results/numbers_seed42',f'{prefix}/results/salve_seed42']
        for seed in SEEDS:
            paths += [f'{prefix}/results/salve_seed{seed}']
        command(['git','add','--',*paths])
        for seed in SEEDS:
            command(['git','add','-f',f'{prefix}/results/salve_seed{seed}/soft_prompt.pt'])
        command(['git','commit','-m','Record four additional fine-tuned-teacher SALVE seeds (43–46)'])
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        command(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'])
        remote=subprocess.check_output(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],cwd=REPO,text=True).split()[0]
        assert remote==sha,'Remote commit does not match; refusing shutdown'
        api=client();pod=api.pod(POD);assert 'stop' in pod.get('actions',[])
        # Credentials are removed before the stop request; API headers remain only in memory.
        for name in ('github_key','runpod_api_key'):(RUNTIME/name).unlink()
        subprocess.run(['sync'],check=True)
        print(f'PUSH VERIFIED {sha}; requesting stop of {POD}',flush=True)
        api.request('POST',f'pods/{POD}/action',{'action':'stop'})
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',failed_unix=time.time());write('status.json',state)
        print('JOB FAILED: pod left running; results remain on the network volume.',flush=True)
        raise
    finally:
        if monitor.poll() is None:monitor.terminate();monitor.wait()
        if not monitor_log.closed:monitor_log.close()

if __name__=='__main__':main()
