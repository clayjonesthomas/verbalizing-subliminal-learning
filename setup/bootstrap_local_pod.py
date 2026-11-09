"""Bootstrap an explicitly allocated pod-local GPU using the shared runner budget.

No network volumes are created. Secrets travel over SSH, never in logs or argv.
"""
import argparse
import base64
import hashlib
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
    parser.add_argument('--direct-model', action='store_true', help='Download pinned model bytes directly from Hugging Face; verify against source hashes')
    parser.add_argument('--finish', action='store_true', help='Verify a completed bootstrap after the startup-command restart; re-arm its existing budget')
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
                   authorization=client.headers['Authorization'],
                   bootstrap_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    code = '''import base64,json,pathlib,sys,tempfile
p=json.load(sys.stdin);d=pathlib.Path(tempfile.mkdtemp(prefix='podrun-bootstrap-'))
bundle=d/'runner.pyz';bundle.write_bytes(base64.b64decode(p['runner']));sys.path.insert(0,str(bundle))
from podrun.budget import arm
out=pathlib.Path(p['config']['run_dir']);out.mkdir(parents=True,exist_ok=True)
if (out/'launch.json').exists() or (out/'state.json').exists():raise RuntimeError('Scientific launch already attempted; do not bootstrap over it')
saved=out/'bootstrap_budget.json';receipt=json.loads(saved.read_text()) if saved.exists() else {}
pid=receipt.get('pid');cmd=pathlib.Path(f'/proc/{pid}/cmdline')
if not (cmd.exists() and b'budget-watchdog' in cmd.read_bytes()):receipt=arm(p['config'],p['authorization'])
receipt['bootstrap_script_sha256']=p['bootstrap_script_sha256']
(out/'bootstrap_budget.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
'''
    run(target, ['python3', '-c', code], input=json.dumps(payload), text=True, timeout=120)
    print('Independent startup budget armed.', flush=True)
    if args.finish:
        if target.get('args') != START or not ready(client.pod(target['id']),timeout=60):
            raise RuntimeError('Project startup or environment is not ready yet')
        print('Project readiness and post-restart budget verified.',flush=True)
        return

    project = '/workspace/projects/verbalizing-subliminal-learning'
    dataset = project+'/experiments/2026-11-06-003-qwen-lls/runs/qwen-lls-p02/artifacts/dataset'
    graphing = project+'/infra-7862638ebfb0ca06c7a0b1c3cd6f0e74e2e2f7d7/graphing-utils'
    key = project+'/.ssh/runpod-caleb-spar-key'
    probe = "import os,json;from pathlib import Path;print(json.dumps(str(Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen3.5-9B')))"
    hf = json.loads(subprocess.check_output(ssh_args(source, ['python','-c',probe]), text=True, timeout=60))
    paths = [dataset, graphing, key, project+'/experiments/'+c['experiment']+'/shared']
    host, port = endpoint(target)
    known = Path('~/.cache/runpod-vsl/known_hosts').expanduser().read_text()
    known_path = '/tmp/pilot-known-hosts-'+target['id']
    run(source, ['python3','-c','import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())',known_path], input=known,text=True)
    transport = shlex.join(['ssh','-p',str(port),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+known_path])
    print('Copying model cache, dataset, graphing code, comparison results and deployment credential over SSH.', flush=True)
    run(source, ['rsync','-a','--relative','-e',transport,*paths,f'root@{host}:/'],timeout=1800)
    # Cache blobs may themselves link outside the model directory. Materialize the exact snapshot.
    snapshot=hf+'/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    model_copy = ['rsync','-aL','--relative','--partial']
    if args.direct_model: model_copy += ['--exclude=*.safetensors','--exclude=.*safetensors*']
    run(source, model_copy+['-e',transport,snapshot,f'root@{host}:/'],timeout=1800)
    manifest_code = "import json,hashlib;from pathlib import Path;p=Path(__SNAPSHOT__);print(json.dumps({x.name:dict(bytes=x.stat().st_size,sha256=hashlib.file_digest(x.open('rb'),'sha256').hexdigest()) for x in p.glob('*.safetensors')}))".replace('__SNAPSHOT__', repr(snapshot))
    manifest = json.loads(subprocess.check_output(raw(source, ['python3','-c',manifest_code]),text=True,timeout=600))
    if len(manifest) != 4 or not all(len(v['sha256']) == 64 for v in manifest.values()):
        raise RuntimeError('Unexpected pinned model shard manifest')
    if args.direct_model:
        download_code = r"""import concurrent.futures,hashlib,json,os,sys,urllib.request
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.load(sys.stdin)
def fetch(item):
 name,expected=item; dest=root/name
 if dest.exists() and dest.stat().st_size==expected['bytes'] and hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==expected['sha256']:
  print('Reused verified shard '+name,flush=True); return
 url='https://huggingface.co/Qwen/Qwen3.5-9B/resolve/c202236235762e1c871ad0ccb60c8ee5ba337b9a/'+name
 tmp=dest.with_name(name+'.download'); digest=hashlib.sha256(); size=0
 with urllib.request.urlopen(url,timeout=120) as response, tmp.open('wb') as out:
  while data:=response.read(4*1024*1024):out.write(data);digest.update(data);size+=len(data)
 if size!=expected['bytes'] or digest.hexdigest()!=expected['sha256']:raise RuntimeError('Downloaded shard differs from pinned source: '+name)
 tmp.replace(dest);print('Downloaded and verified '+name,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,manifest.items()))
"""
        run(target,['python3','-c',download_code,snapshot],input=json.dumps(manifest),text=True,timeout=1800)
    verify_code = "import json,hashlib,sys;from pathlib import Path;p=Path(sys.argv[1]);expected=json.load(sys.stdin);actual={x.name:dict(bytes=x.stat().st_size,sha256=hashlib.file_digest(x.open('rb'),'sha256').hexdigest()) for x in p.glob('*.safetensors')};assert actual==expected,'Pinned model shard verification failed';(p/'pilot_input_verification.json').write_text(json.dumps(actual,indent=2));print('Verified all four pinned model shards',flush=True)"
    run(target,['python3','-c',verify_code,snapshot],input=json.dumps(manifest),text=True,timeout=600)
    run(source, ['rsync','-a','-e',transport,REPO+'/setup/github_known_hosts',f'root@{host}:{project}/github_known_hosts'],timeout=60)
    runtime_key='/run/runpod-caleb-spar/runpod-caleb-spar-key'
    run(target, ['install','-d','-m','700','/run/runpod-caleb-spar'])
    run(target, ['install','-m','600',key,runtime_key])
    git_ssh=shlex.join(['ssh','-i',runtime_key,'-o','IdentitiesOnly=yes','-o','IdentityAgent=none','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+project+'/github_known_hosts'])
    exists=subprocess.run(raw(target,['test','-d',REPO+'/.git'])).returncode==0
    if not exists:
        run(target, ['env','GIT_SSH_COMMAND='+git_ssh,'git','clone','--filter=blob:none','--depth','1','--single-branch','git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git',REPO],timeout=1200)
    run(target, ['git','-C',REPO,'config','core.sshCommand',git_ssh])
    run(target, ['git','-C',REPO,'fetch','origin',c['source_commit']],timeout=600)
    run(target, ['git','-C',REPO,'checkout','--detach',c['source_commit']],timeout=600)
    run(target, ['env','UV_HTTP_TIMEOUT=180','UV_HTTP_RETRIES=5','bash',REPO+'/setup/boot.sh'],timeout=1200)
    if client.pod(target['id']).get('args') != START:
        client.request('PATCH','pods/'+target['id'],{'args':START})
    for attempt in range(120):
        target=client.pod(target['id'])
        if ready(target,timeout=60):break
        time.sleep(5)
    else:raise RuntimeError('Bootstrap finished but project readiness has not been verified')
    # Updating startup configuration may replace the container and its watchdog.
    target = client.pod(target['id'])
    run(target, ['python3','-c',code],input=json.dumps(payload),text=True,timeout=120)
    print('Project Python/CUDA and future startup verified; ready for shared-runner launch.',flush=True)


if __name__ == '__main__':
    main()
