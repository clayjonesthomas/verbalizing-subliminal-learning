#!/usr/bin/env python3
"""Project-scoped Runpod access. Run on the laptop; never copy its API key."""
import argparse
import ipaddress
import json
from pathlib import Path
import re
import shlex
import ssl
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

TEAM = 'cmu31hyiy002gk9s0xik85gob'
ACCOUNT = 'user_3JNPrVMTYwugXaYJytSv2t8lRsv'
KEY_FILE = Path('~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key')
POD = 'dsln6o9uc00dwc'
VOLUME = 'ae6w0yoyy6'
REPO = '/workspace/projects/verbalizing-subliminal-learning/repo'
IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
START = f'bash {REPO}/setup/start.sh'


class Client:
    def __init__(self):
        self.headers = {
            'Authorization': 'Bearer ' + KEY_FILE.expanduser().read_text().strip(),
            'X-Team-Id': TEAM,
            'Content-Type': 'application/json',
            'User-Agent': 'vsl-runpod-setup/1.0',
        }
        pem = Path('/etc/ssl/cert.pem')
        self.context = ssl.create_default_context(cafile=str(pem) if pem.exists() else None)

    def request(self, method, path, body=None):
        url = ('https://api.runpod.io/graphql' if path == 'graphql'
               else 'https://api.runpod.io/v2/' + path)
        request = Request(url, headers=self.headers, method=method,
                          data=json.dumps(body).encode() if body is not None else None)
        try:
            with urlopen(request, context=self.context, timeout=60) as response:
                data = response.read()
                return json.loads(data) if data else {}
        except HTTPError as exc:
            # Never echo response bodies, request headers, or pod environments.
            raise RuntimeError(f'Runpod {method} {path}: HTTP {exc.code}; no automatic retry') from None

    def verify(self):
        result = self.request('POST', 'graphql', {'query': '{ myself { id email } }'})
        if result.get('data', {}).get('myself', {}).get('id') != ACCOUNT:
            raise RuntimeError('Caleb SPAR account verification failed; refusing to proceed')

    def pod(self, pod_id):
        self.verify()
        pod = self.request('GET', f'pods/{pod_id}')
        if pod.get('id') != pod_id or not any(
            m.get('volumeId') == VOLUME and m.get('path') == '/workspace'
            for m in pod.get('mounts', {}).get('network', [])
        ):
            raise RuntimeError('Pod does not mount this project volume at /workspace')
        # v2 occasionally omits ssh.direct even while the pod is running.
        # Resolve the public port via the same team-scoped GraphQL API then.
        if pod.get('status') == 'RUNNING' and not (pod.get('ssh') or {}).get('direct'):
            query = ('{ pod(input: {podId: ' + json.dumps(pod_id) + '}) '
                     '{ id runtime { ports { ip isIpPublic privatePort publicPort type } } } }')
            data = self.request('POST', 'graphql', {'query': query})
            detail = data.get('data', {}).get('pod') or {}
            if detail.get('id') != pod_id:
                raise RuntimeError('SSH endpoint lookup returned a different pod')
            ports = (detail.get('runtime') or {}).get('ports', [])
            matches = [p for p in ports if p.get('isIpPublic') and p.get('privatePort') == 22
                       and p.get('type') == 'tcp']
            if len(matches) == 1:
                pod['ssh'] = {'direct': {'host': matches[0]['ip'],
                              'port': matches[0]['publicPort'], 'username': 'root'}}
        return pod


def endpoint(pod):
    if pod.get('status') != 'RUNNING':
        raise RuntimeError('Pod is not running')
    direct = (pod.get('ssh') or {}).get('direct') or {}
    if not direct.get('host'):
        raise RuntimeError('Direct SSH is not ready; retry when the pod finishes starting')
    host = str(ipaddress.ip_address(direct['host']))
    port = int(direct.get('port', 0))
    if not 1 <= port <= 65535 or direct.get('username') != 'root':
        raise RuntimeError('No valid direct root SSH endpoint')
    return host, port


def ssh_args(pod, command=None):
    host, port = endpoint(pod)
    known_hosts = Path('~/.cache/runpod-vsl/known_hosts').expanduser()
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    args = ['ssh', '-A', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', f'UserKnownHostsFile={known_hosts}', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '-p', str(port)]
    if command is None:
        args += ['-t', f'root@{host}', f'cd {shlex.quote(REPO)} && exec bash -l']
    else:
        args += [f'root@{host}', shlex.join(['bash', f'{REPO}/setup/run.sh', *command])]
    return args


def ssh(pod, command=None):
    subprocess.run(ssh_args(pod, command), check=True)


def ready(pod, timeout=30):
    if pod.get('status') != 'RUNNING' or not (pod.get('ssh') or {}).get('direct'):
        return False
    check = ['python', '-c',
             'import sys, torch; '
             'assert sys.prefix == "/opt/verbalizing-subliminal-learning-venv"; '
             'assert torch.cuda.is_available(); '
             'assert torch.ones(1, device="cuda").item() == 1']
    try:
        result = subprocess.run(ssh_args(pod, check), capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    if result.returncode == 0:
        return True
    # Retry transport startup only. Never suppress bad credentials or changed keys.
    if result.returncode == 255 and any(message in result.stderr.lower() for message in (
        'connection refused', 'connection timed out', 'operation timed out',
        'connection reset', 'connection closed', 'no route to host',
    )):
        return False
    raise RuntimeError('SSH authentication/host verification or environment readiness check failed. '
                       'Inspect the pod startup logs and SSH credentials; no command was run.')


def start_and_connect(client, pod, command=None, timeout=900):
    deadline = time.monotonic() + timeout
    status = pod.get('status')
    if status not in ('RUNNING', 'STARTING', 'PROVISIONING'):
        if 'start' not in pod.get('actions', []):
            raise RuntimeError(f'Runpod does not offer start for this pod ({status})')
        if pod.get('image') != IMAGE or pod.get('args') != START:
            raise RuntimeError('Automatic project startup is not configured for this pod')
        # Send once only. Ambiguous API failures must not cause repeated mutations.
        client.verify()
        client.request('POST', f"pods/{pod['id']}/action", {'action': 'start'})
        print('Start requested. Waiting for project restore and SSH...', file=sys.stderr, flush=True)
        pod = client.pod(pod['id'])
    else:
        print('Checking pod readiness...', file=sys.stderr, flush=True)
    while time.monotonic() < deadline:
        if pod.get('status') in ('ERROR', 'TERMINATED'):
            raise RuntimeError(f"Pod entered {pod['status']} during startup; inspect Runpod logs")
        if ready(pod, timeout=min(30, max(1, deadline - time.monotonic()))):
            print('Pod ready. Connecting...', file=sys.stderr, flush=True)
            # Execute the user's command exactly once; never retry its side effects.
            ssh(pod, command)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print('Waiting for project restore and SSH...', file=sys.stderr, flush=True)
        time.sleep(min(5, remaining))
        pod = client.pod(pod['id'])
    raise RuntimeError('Timed out waiting for the pod. It has not been stopped; '
                       'inspect status/startup logs or rerun start-and-connect.')


def save_and_stop(client, pod, dry_run=False):
    if 'stop' not in pod.get('actions', []):
        raise RuntimeError('Runpod does not currently offer stop for this pod; no termination fallback')
    ssh(pod, ['bash', 'setup/save.sh'])  # Any failure prevents the API action.
    fresh = client.pod(pod['id'])
    if (endpoint(fresh) != endpoint(pod) or fresh.get('startedAt') != pod.get('startedAt')
            or (fresh.get('runtime') or {}).get('uptime', 0) < (pod.get('runtime') or {}).get('uptime', 0)
            or 'stop' not in fresh.get('actions', [])):
        raise RuntimeError('Pod changed during save; refusing to stop it')
    if dry_run:
        print('Snapshot saved and pod reverified. Dry run: no stop request sent.')
        return
    result = client.request('POST', f"pods/{pod['id']}/action", {'action': 'stop'})
    print(json.dumps({'id': pod['id'], 'action': 'stop', 'status': result.get('status')}))


def enable_startup(client, pod, dry_run=False):
    if pod.get('image') != IMAGE or pod.get('args') not in ('', START):
        raise RuntimeError('Unexpected image/start command; refusing to overwrite custom startup')
    if pod.get('args') == START:
        print('Automatic startup is already configured.')
        return
    if dry_run:
        print(json.dumps({'id': pod['id'], 'args': START}))
        return
    # Configuration changes can recreate the container. Save first, without stopping.
    ssh(pod, ['bash', '-n', 'setup/start.sh'])
    ssh(pod, ['bash', 'setup/save.sh'])
    fresh = client.pod(pod['id'])
    if (fresh.get('args') != pod.get('args') or fresh.get('image') != IMAGE
            or fresh.get('startedAt') != pod.get('startedAt')):
        raise RuntimeError('Pod changed during save; refusing to change startup')
    client.request('PATCH', f"pods/{pod['id']}", {'args': START})
    if client.pod(pod['id']).get('args') != START:
        raise RuntimeError('Startup update could not be verified')
    print('Automatic startup configured; original /start.sh follows project restore.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod', default=POD, help='Replacement pod ID; must mount this same volume')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('status')
    access = sub.add_parser('start-and-connect', help='Start if needed, wait for readiness, then connect with agent forwarding')
    access.add_argument('--timeout', type=int, default=900, help='Readiness wait in seconds (default: 900)')
    access.add_argument('command', nargs=argparse.REMAINDER)
    sub.add_parser('save')
    for action in ('save-and-stop', 'enable-startup'):
        sub.add_parser(action).add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.pod):
        parser.error('Invalid pod ID')
    if args.action == 'start-and-connect' and args.timeout <= 0:
        parser.error('--timeout must be positive')
    client = Client()
    pod = client.pod(args.pod)
    if args.action == 'status':
        print(json.dumps({k: pod.get(k) for k in ('id', 'name', 'status', 'image', 'args', 'mounts', 'ssh')}, indent=2))
    elif args.action == 'start-and-connect':
        command = args.command
        if command[:1] == ['--']:
            command = command[1:]
        start_and_connect(client, pod, command or None, args.timeout)
    elif args.action == 'save':
        ssh(pod, ['bash', 'setup/save.sh'])
    elif args.action == 'save-and-stop':
        save_and_stop(client, pod, args.dry_run)
    elif args.action == 'enable-startup':
        enable_startup(client, pod, args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        print(f'Remote command failed (exit {exc.returncode}); no further action taken.', file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
