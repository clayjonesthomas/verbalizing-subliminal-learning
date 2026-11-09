"""Bootstrap an explicitly allocated pod-local GPU using the shared runner budget.

No network volumes are created. Secrets travel over SSH, never in logs or argv.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from runpod import Client, REPO, START, endpoint, ssh_args, ready


def raw(pod, command):
    args = ssh_args(pod, ['true'])
    args[-1] = shlex.join(command)
    return args


def run(pod, command, **kwargs):
    return subprocess.run(raw(pod, command), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--source-pod', required=True)
    parser.add_argument('--volume-gb', type=int, default=120)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    client = Client(local_volume_gb=args.volume_gb)
    target = client.pod(c['provider']['pod_id'])
    source = Client().pod(args.source_pod)
    # Only readiness transport is retried. No resource creation is performed here.
    for attempt in range(60):
        target = client.pod(target['id'])
        if (target.get('ssh') or {}).get('direct'):
            probe = subprocess.run(raw(target, ['true']), capture_output=True)
            if probe.returncode == 0:
                break
            if b'Permission denied' in probe.stderr or b'HOST IDENTIFICATION HAS CHANGED' in probe.stderr:
                raise RuntimeError('SSH identity verification failed')
        time.sleep(5)
    else:
        raise RuntimeError('SSH not ready; reconcile allocated pod before retrying')

    infra = Path(os.environ.get('RESEARCH_INFRA_DIR', '~/code/research-infra')).expanduser()
    sys.path.insert(0, str(infra/'experiment-runner'))
    from podrun.cli import bundle_runner
    from podrun.budget import configure
    bootstrap = dict(c)
    bootstrap.pop('failure_hold_seconds', None)  # The scientific launch claims its own lease later.
    bootstrap['run_dir'] = f"/workspace/projects/verbalizing-subliminal-learning/experiments/{c['experiment']}/runs/{c['run_id']}"
    bootstrap = configure(bootstrap)
    payload = dict(config=bootstrap, runner=base64.b64encode(bundle_runner()).decode(),
                   authorization=client.headers['Authorization'])
    code = '''import base64,json,pathlib,sys,tempfile
p=json.load(sys.stdin);d=pathlib.Path(tempfile.mkdtemp(prefix='podrun-bootstrap-'))
bundle=d/'runner.pyz';bundle.write_bytes(base64.b64decode(p['runner']));sys.path.insert(0,str(bundle))
from podrun.budget import arm
receipt=arm(p['config'],p['authorization'])
out=pathlib.Path(p['config']['run_dir']);out.mkdir(parents=True,exist_ok=True)
(out/'bootstrap_budget.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
'''
    run(target, ['python3', '-c', code], input=json.dumps(payload), text=True, timeout=120)
    del payload
    print('Independent startup budget armed.', flush=True)

    project = '/workspace/projects/verbalizing-subliminal-learning'
    dataset = project+'/experiments/2026-11-06-003-qwen-lls/runs/qwen-lls-p02/artifacts/dataset'
    graphing = project+'/infra-7862638ebfb0ca06c7a0b1c3cd6f0e74e2e2f7d7/graphing-utils'
    key = project+'/.ssh/runpod-caleb-spar-key'
    probe = "import os,json;from pathlib import Path;print(json.dumps(str(Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen3.5-9B')))"
    hf = json.loads(subprocess.check_output(ssh_args(source, ['python','-c',probe]), text=True, timeout=60))
    paths = [hf, dataset, graphing, key, project+'/experiments/'+c['experiment']+'/shared']
    host, port = endpoint(target)
    known = Path('~/.cache/runpod-vsl/known_hosts').expanduser().read_text()
    known_path = '/tmp/pilot-known-hosts-'+target['id']
    run(source, ['python3','-c','import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())',known_path], input=known,text=True)
    transport = shlex.join(['ssh','-p',str(port),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+known_path])
    print('Copying model cache, dataset, graphing code, comparison results and deployment credential over SSH.', flush=True)
    run(source, ['rsync','-a','--relative','-e',transport,*paths,f'root@{host}:/'],timeout=1800)
    run(source, ['rsync','-a','-e',transport,REPO+'/setup/github_known_hosts',f'root@{host}:{project}/github_known_hosts'],timeout=60)
    git_ssh=shlex.join(['ssh','-i',key,'-o','IdentitiesOnly=yes','-o','IdentityAgent=none','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+project+'/github_known_hosts'])
    run(target, ['env','GIT_SSH_COMMAND='+git_ssh,'git','clone','--filter=blob:none','--depth','1','--single-branch','git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git',REPO],timeout=1200)
    run(target, ['git','-C',REPO,'config','core.sshCommand',git_ssh])
    run(target, ['git','-C',REPO,'fetch','origin',c['source_commit']],timeout=600)
    run(target, ['git','-C',REPO,'checkout','--detach',c['source_commit']],timeout=600)
    run(target, ['env','UV_HTTP_TIMEOUT=180','UV_HTTP_RETRIES=5','bash',REPO+'/setup/boot.sh'],timeout=1200)
    client.request('PATCH','pods/'+target['id'],{'args':START})
    if not ready(client.pod(target['id']),timeout=60):
        raise RuntimeError('Bootstrap finished but project readiness has not been verified')
    print('Project Python/CUDA and future startup verified; ready for shared-runner launch.',flush=True)


if __name__ == '__main__':
    main()
