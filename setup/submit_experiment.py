"""Scripted submission adapter; no agent polling or duplicate paid retries."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def materialize(template, root, source, infra_sha, run_id, pod_id):
    c = copy.deepcopy(template)
    c.update(run_id=run_id, source_commit=source, infrastructure_commit=infra_sha)
    c.pop('run_dir', None)
    for key in ('provider', 'laptop_provider'):
        c[key]['pod_id'] = pod_id
    c.setdefault('workspace', {}).setdefault('repo', str(root))
    # Historical templates use a bare RUNTIME only for the pull hook's run ID.
    c['workspace']['pull'] = [run_id if v == 'RUNTIME' else v for v in c['workspace'].get('pull', [])]
    return c


def remote(provider, endpoint, script):
    from podrun.watcher import ssh_args
    identity = "import pathlib\ne=dict(x.split(b'=',1) for x in pathlib.Path('/proc/1/environ').read_bytes().split(b'\\0') if b'=' in x)\nassert e.get(b'RUNPOD_POD_ID',b'').decode()==" + repr(provider['pod_id']) + "\n"
    result = subprocess.run(ssh_args(provider, endpoint) + [shlex.join(['python3', '-c', identity + script])],
                            text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError('Submission SSH check failed; inspect submission log/pod, no automatic reallocation')
    return json.loads(result.stdout)


def verify(c, endpoint, timeout=180):
    directory = repr(c['run_dir'])
    script = """import json,pathlib,os
p=pathlib.Path(DIRECTORY)
def read(name):
 try: return json.loads((p/name).read_text())
 except FileNotFoundError: return {}
s=read('state.json');g=read('guard.json');b=s.get('budget_watchdog') or {}
assert not s or (s.get('run_id')==RUN_ID and s.get('source_commit')==SOURCE_COMMIT), 'Wrong run provenance'
def live(pid):
 try: os.kill(int(pid or 0),0); return bool(pid)
 except (ProcessLookupError,PermissionError,ValueError): return False
print(json.dumps(dict(status=s.get('status'),outcome=s.get('outcome'),ready=s.get('status')=='running' and g.get('status')=='running' and g.get('supervisor_pid')==s.get('pid') and b.get('ready') is True and all(live(pid) for pid in (s.get('pid'),g.get('pid'),b.get('pid'))),budget_ready=b.get('ready') is True)))
""".replace('DIRECTORY', directory).replace('RUN_ID', repr(c['run_id'])).replace('SOURCE_COMMIT', repr(c['source_commit']))
    deadline = time.monotonic() + timeout
    while True:
        evidence = remote(c['laptop_provider'], endpoint, script)
        if evidence['ready']:
            return evidence
        # Short jobs may finish before polling: retain an honest terminal receipt.
        if evidence.get('outcome') == 'success' and evidence['budget_ready']:
            return evidence
        if evidence['status'] in ('held', 'supervisor_failed', 'supervisor_lost', 'cleanup_failed', 'termination_unknown'):
            raise RuntimeError('Worker stopped before readiness; inspect shared runner snapshot')
        if time.monotonic() >= deadline:
            raise RuntimeError('Readiness unconfirmed; resume verifies only, never relaunches')
        time.sleep(3)


def budget_check(c, endpoint):
    b = c['budget']
    script = """import json,pathlib,time
p=pathlib.Path(DIRECTORY)/'ledger.json'
if not p.exists(): print(json.dumps({'exists':False}))
else:
 d=json.loads(p.read_text())
 total=d['prior_spend_usd']+sum(max(0,(x.get('stopped_at') or time.time())-x['started_at'])*x['hourly_usd']/3600 for x in d['pods'].values())
 print(json.dumps({'exists':True,'total':total,'limit':d['limit_usd'],'identity':d['identity'],'baseline':d['prior_spend_usd'],'stopped':d.get('stop_requested',False)}))
""".replace('DIRECTORY', repr(b['directory']))
    result = remote(c['laptop_provider'], endpoint, script)
    if result.get('exists') and (result['stopped'] or result['baseline'] != b['prior_spend_usd'] or result['identity'] != b['identity'] or result['limit'] != b['limit_usd'] or result['total'] >= b['limit_usd']):
        raise RuntimeError('Shared budget exhausted or mismatched; refusing further allocation/launch')


def run_plan(plan, root, infra, directory):
    from podrun.cli import preflight, launch
    from podrun.common import atomic, lock
    from podrun.submission import execute
    from runpod import Client, endpoint, start_and_connect
    from provision import provision, provisioning_lock
    from gpu_demand import DemandHeartbeat
    os.environ['PODRUN_WORKSPACE_REPO'] = str(root)
    # One flow per experiment, including distinct submission IDs and all agents.
    with lock(directory.parent / 'experiment.lock'):
        def pinned():
            if git(root, 'rev-parse', 'HEAD') != plan['source_commit'] or git(infra, 'rev-parse', 'HEAD') != plan['infrastructure_commit'] or git(infra, 'status', '--porcelain', '--', 'experiment-runner'):
                raise RuntimeError('Pinned controller checkout changed; resume from original clean checkouts')
        pinned()
        configs = {w['run_id']: preflight(w['config'], forward_env=plan['forward_env']) for w in plan['workers']}
        client = Client()
        endpoints = {}
        last = [None]
        def config(worker, slot):
            c = copy.deepcopy(configs[worker['run_id']])
            for key in ('provider', 'laptop_provider'):
                c[key]['pod_id'] = slot['pod_id']
            return c
        def allocate(worker, transition):
            args = argparse.Namespace(**plan['placement'], name=worker['run_id'], pod_local_gb=None)
            with DemandHeartbeat(allow_discovery=False) as demand:
                with provisioning_lock():
                    demand.enable_discovery()
                    if last[0] is None:
                        client.verify()
                        inventory = client.request('GET', 'pods')
                        pods = inventory if isinstance(inventory, list) else inventory.get('pods', [])
                        probe = configs[worker['run_id']]
                        for pod in pods:
                            if pod.get('status') == 'RUNNING' and any(m.get('volumeId') == probe['provider']['volume_id'] and m.get('path') == '/workspace' for m in pod.get('mounts', {}).get('network', [])):
                                ep = endpoint(client.pod(pod['id']))
                                probe = copy.deepcopy(probe)
                                probe['laptop_provider']['pod_id'] = pod['id']
                                budget_check(probe, ep)
                                last[0] = (probe, ep)
                                break
                    def checked(state, **fields):
                        if state == 'allocating':
                            pinned()
                            if last[0]:
                                prior, ep = last[0]
                                budget_check(prior, ep)
                        transition(state, **fields)
                    pod = provision(args, demand=demand, transition=checked)
                    endpoints[pod['id']] = endpoint(client.pod(pod['id']))
        def ready(worker, slot):
            if slot['pod_id'] not in endpoints:
                pod = client.pod(slot['pod_id'])
                start_and_connect(client, pod, ['true'])
                endpoints[slot['pod_id']] = endpoint(client.pod(slot['pod_id']))
            c = config(worker, slot)
            budget_check(c, endpoints[slot['pod_id']])
            last[0] = (c, endpoints[slot['pod_id']])
        def start(worker, slot):
            pinned()
            c = config(worker, slot)
            path = directory / (worker['run_id'] + '.json')
            atomic(path, c)
            launch(c, False, forward_credentials=True, config_path=path, forward_env=plan['forward_env'], submission=True)
        def check(worker, slot):
            c = config(worker, slot)
            ep = endpoints.get(slot['pod_id']) or endpoint(client.pod(slot['pod_id']))
            result = verify(c, ep)
            last[0] = (c, ep)
            return result
        return execute(directory / 'receipt.json', plan['workers'], allocate, ready, start, check)


def main(root, infra, argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--template', type=Path)
    p.add_argument('--pod', action='append', default=[], help='Existing ready/held pod, one per worker; same experiment budget only')
    p.add_argument('--gpu', action='append')
    p.add_argument('--allow-non-rtx', action='store_true')
    p.add_argument('--max-hourly', type=float, default=5)
    p.add_argument('--forward-env', action='append', default=[])
    p.add_argument('--submission', help='Resume the exact stored submission; ambiguous outcomes are verification-only')
    p.add_argument('--foreground', action='store_true', help='Run deterministic controller in this terminal')
    p.add_argument('--status', action='store_true')
    a = p.parse_args(argv)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}-\d{3}-[a-z0-9-]+', a.experiment):
        p.error('Invalid experiment ID')
    if not 1 <= a.workers <= 32 or (a.pod and len(a.pod) != a.workers) or len(set(a.pod)) != len(a.pod):
        p.error('Use 1–32 workers and one distinct --pod per worker when reusing pods')
    if not 0 < a.max_hourly <= 10 or (a.allow_non_rtx and not a.gpu):
        p.error('Invalid price limit or non-RTX selection')
    from runpod import TEAM, ACCOUNT, VOLUME
    from provision import preferred
    if a.gpu and not a.allow_non_rtx and any(not preferred({'id': g}) for g in a.gpu):
        p.error('Non-RTX GPUs require explicit approved --allow-non-rtx selection')
    experiment = root / 'fork_experiments' / a.experiment
    sid = a.submission or ('submit-' + uuid.uuid4().hex[:12])
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', sid):
        p.error('Invalid submission ID')
    directory = experiment / 'results' / 'submissions' / sid
    if a.status:
        if not a.submission: p.error('--status requires --submission ID')
        for name in ('controller.json', 'receipt.json', 'failure.json'):
            if (directory / name).exists(): print((directory / name).read_text())
        return 0
    from podrun.common import atomic
    from podrun.cli import preflight, forwarded_environment
    if a.submission:
        plan = json.loads((directory / 'plan.json').read_text())
        if plan['source_commit'] != git(root, 'rev-parse', 'HEAD') or plan['infrastructure_commit'] != git(infra, 'rev-parse', 'HEAD'):
            raise ValueError('Resume must use original pinned source and infrastructure checkout')
    else:
        template_path = (a.template or experiment / 'launch.template.json').resolve()
        relative = str(template_path.relative_to(root))
        source, infra_sha = git(root, 'rev-parse', 'HEAD'), git(infra, 'rev-parse', 'HEAD')
        # Read committed template, not potentially dirty local configuration.
        template = json.loads(git(root, 'show', source + ':' + relative))
        if template.get('experiment') != a.experiment:
            raise ValueError('Template experiment does not match submission')
        paths = ['setup', relative, *template.get('checkout', {}).get('code_paths', [])]
        if git(root, 'status', '--porcelain', '--', *paths):
            raise ValueError('Commit relevant code/config before submitting')
        if git(infra, 'status', '--porcelain', '--', 'experiment-runner'):
            raise ValueError('Commit shared runner before submitting')
        for repo, sha in ((root, source), (infra, infra_sha)):
            if not git(repo, 'branch', '-r', '--contains', sha):
                raise ValueError('Push pinned commit before submitting')
        policy = experiment / 'report_policy.json'
        if policy.exists() and json.loads(policy.read_text()).get('shared_graphing_required'):
            sys.path.insert(0, str(infra / 'graphing-utils'))
            from research_graphing.validation import check_report
            errors = check_report(experiment)
            if errors: raise ValueError('Report preflight failed: ' + '; '.join(errors))
        workers = []
        for i in range(a.workers):
            rid = sid + '-w' + str(i + 1)
            pod = a.pod[i] if a.pod else None
            c = materialize(template, root, source, infra_sha, rid, pod or 'PENDING')
            for key in ('provider', 'laptop_provider'):
                if any(c[key].get(k) != v for k, v in dict(team_id=TEAM, account_id=ACCOUNT, volume_id=VOLUME, mount_path='/workspace').items()) or c[key].get('storage_mode') == 'pod_local':
                    raise ValueError('Submission requires the project shared-volume scope')
            preflight(c, forward_env=a.forward_env)
            workers.append(dict(run_id=rid, pod_id=pod, config=c))
        plan = dict(schema=1, source_commit=source, infrastructure_commit=infra_sha, workers=workers,
                    forward_env=a.forward_env, placement=dict(gpu=a.gpu, allow_non_rtx=a.allow_non_rtx, max_hourly=a.max_hourly))
        atomic(directory / 'plan.json', plan)
    forwarded_environment(plan['forward_env'])
    if not a.foreground:
        command = [sys.executable, str(root / 'setup/experiment.py'), 'submit', '--experiment', a.experiment,
                   '--submission', sid, '--foreground']
        with (directory / 'controller.log').open('ab') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, env=dict(os.environ, RESEARCH_INFRA_DIR=str(infra)))
        atomic(directory / 'controller.json', dict(pid=child.pid, submission=sid, log=str(directory / 'controller.log')))
        print(json.dumps(dict(submission=sid, pid=child.pid, receipt=str(directory / 'receipt.json'), log=str(directory / 'controller.log'))))
        return 0
    try:
        result = run_plan(plan, root, infra, directory)
        print(json.dumps(result))
    except BaseException as exc:
        # Exception messages from external APIs can contain credentials; only type
        # and safe recovery instructions belong in durable diagnostics.
        atomic(directory / 'failure.json', dict(error_type=type(exc).__name__, action='Inspect receipt and pod. Resume never repeats an uncertain create or launch.'))
        print('Submission stopped: ' + type(exc).__name__ + '; inspect receipt and shared-runner snapshot.', file=sys.stderr)
        return 1
    return 0
