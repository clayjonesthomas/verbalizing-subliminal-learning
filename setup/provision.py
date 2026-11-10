#!/usr/bin/env python3
"""Prefer RTX PRO 6000 GPUs >=48 GB and <$5/hour on the US-NE-1 volume.

Run explicitly to create ONE new paid pod. Ctrl-C stops searching. Uncertain
creation outcomes stop the script: inspect the account before running it again.
The container runs project startup automatically; success waits for SSH/CUDA.
Only RTX PRO 6000 variants are eligible; capacity rejections retry until stopped.
Concurrent invocations share two allocation slots; additional invocations wait.
"""
import json
from contextlib import contextmanager, nullcontext
import argparse
import fcntl
import os
from datetime import datetime, timezone
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path
from runpod import ACCOUNT, TEAM, Client, IMAGE, START, VOLUME, start_and_connect, runpod_http, RateLimitError, rate_limit_delay


BILLING_LOG = Path('~/code/personal-scripts-and-config/logs/runpod-billing.jsonl').expanduser()
# Explicit attribution for this laptop's project provisioning, matching rr's
# configured owner. This is billing attribution, not GraphQL's team account ID.
BILLING_ACTOR = 'user_3JNicRGW4EgAlbDQCvV4jDCjD2Q'


def record_billing(event, **fields):
    """Use the same durable ledger as rr runpod-spend; never log pod env/keys."""
    record = dict(schema_version=1, event=event, team_id=TEAM, account_id=ACCOUNT,
                  actor_id=BILLING_ACTOR, attribution_source='project_local_provisioner',
                  time=datetime.now(timezone.utc).isoformat(),
                  source='vsl_provision', **fields)
    BILLING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BILLING_LOG.open('a', encoding='utf-8') as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        log.write(json.dumps(record, sort_keys=True) + '\n')
        log.flush()
        os.fsync(log.fileno())


def create_with_billing(client, body, gpu, allocated=None):
    # Persist requested prices first, and refuse creation if the ledger fails.
    # This also retains reconciliation evidence for an ambiguous API response.
    record_billing('pod_create_requested', name=body['name'],
                   gpu_id=gpu['id'], catalog_gpu_hourly=gpu['price']['secure'],
                   disk_gb=body['disk'])
    pod = client.create(body)
    if allocated is not None:
        allocated(pod)
    # A logging failure after creation propagates to the no-retry path.
    record_billing('pod_created', pod_id=pod['id'], name=body['name'],
                   gpu_id=gpu['id'], catalog_gpu_hourly=gpu['price']['secure'],
                   hourly=pod.get('cost'), disk_gb=body['disk'],
                   created_at=pod.get('createdAt'),
                   pricing_note='API creation cost; catalog fallback if absent. '
                                'Container storage estimated at $0.10/GB/month, 730 hours/month.')
    return pod


class ApiError(RuntimeError):
    def __init__(self, status, capacity=False, reason="", retry_after=None):
        self.retry_after = retry_after
        self.status, self.capacity, self.reason = status, capacity, reason
        super().__init__(f'Runpod creation rejected: HTTP {status}')


class ProvisionClient(Client):
    def create(self, body):
        request = Request('https://api.runpod.io/v2/pods', headers=self.headers,
                          method='POST', data=json.dumps(body).encode())
        try:
            with runpod_http.open_request(request, context=self.context, timeout=60, opener=urlopen, caller="project-provision") as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get('detail', '')
            except (ValueError, AttributeError):
                detail = ''
            raise ApiError(exc.code, exc.code == 400 and
                           any(message in str(detail).lower() for message in ('no longer any instances', 'does not have the resources to deploy your pod')), str(detail)[:500], exc.headers.get("Retry-After")) from None


def preferred(gpu):
    return 'RTX PRO 6000' in gpu.get('id', '').upper()


def candidates(gpus, gpu_ids=None, allow_non_rtx=False, max_hourly=5, min_vram_gb=48):
    return sorted(
        (g for g in gpus if (preferred(g) or (allow_non_rtx and gpu_ids)) and (not gpu_ids or g.get('id') in gpu_ids)
         and g.get('memory', 0) >= min_vram_gb
         and g.get('manufacturer') == 'NVIDIA' and g.get('secure')
         and 0 < (g.get('price', {}).get('secure') or 0)
         and (max_hourly is None or g['price']['secure'] < max_hourly)),
        key=lambda g: (not preferred(g),
                       g.get('memory', 0) < 90 if preferred(g) else False,
                       g['price']['secure']))


PROVISION_LOCK = Path('~/.cache/runpod-vsl/provision.lock').expanduser()


@contextmanager
def provisioning_lock(path=PROVISION_LOCK):
    """Allow two provisioning flows, counting an existing single-lock process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = [path, path.with_name(path.name + '.slot2')]
    print(json.dumps({'event':'waiting_for_provisioning_slot'}), flush=True)
    handle = None
    try:
        while handle is None:
            for slot, candidate in enumerate(paths, 1):
                opened = candidate.open('a')
                try:
                    fcntl.flock(opened, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    opened.close()
                except BaseException:
                    opened.close()
                    raise
                else:
                    handle = opened
                    break
            if handle is None:
                time.sleep(.2)
        print(json.dumps({'event':'provisioning_slot_acquired','slot':slot}), flush=True)
        yield
    finally:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def try_later(operation):
    """Retry read-only setup calls only after the announced quota reset."""
    while True:
        try:
            return operation()
        except RateLimitError as exc:
            print(f'Rate limited; try later in {exc.retry_after:.0f} seconds.', flush=True)
            time.sleep(exc.retry_after)


def provision_cpu(args, demand=None, transition=None):
    c=ProvisionClient();try_later(c.verify)
    volume=try_later(lambda:c.request('GET',f'network-volumes/{VOLUME}'))
    assert volume['dataCenter']=='US-NE-1'
    public_key=Path('~/.ssh/id_ed25519.pub').expanduser().read_text().strip()
    while True:
        catalog=try_later(lambda:c.request('GET','catalog/cpus?include=AVAILABILITY&product=POD&cloud=SECURE'))['cpus']
        options=sorted((x for x in catalog if x['id'] in ['cpu3c','cpu3g'] and x['vcpu']['min']<=2 and 0<x['price']['securePerVcpu']*2<=min(args.max_hourly,.08)),key=lambda x:x['price']['securePerVcpu'])
        if not options:raise RuntimeError('No eligible CPU catalog entry')
        for cpu in options:
            body=dict(name=args.name,image='ubuntu:24.04',args='bash /workspace/projects/verbalizing-subliminal-learning/repo/setup/cpu-boot.sh',cloud='SECURE',cpu=dict(id=cpu['id'],vcpuCount=2),disk=60,dataCenterIds=['US-NE-1'],mounts={'network':[{'volumeId':VOLUME,'path':'/workspace'}]},startSsh=True,ports=['22/tcp'],env={'PUBLIC_KEY':public_key})
            try:
                transition('allocating',pod_name=body['name'])
                pod=create_with_billing(c,body,dict(id=cpu['id'],price={'secure':cpu['price']['securePerVcpu']*2}),allocated=lambda p:transition('allocated',pod_id=p['id']))
            except ApiError as exc:
                if exc.capacity:
                    transition('planned');continue
                if exc.status==429:
                    transition('planned');time.sleep(rate_limit_delay(c.headers,'rest-v2',exc.retry_after));break
                raise
            else:
                if demand:demand.stop()
                print(json.dumps(dict(created=pod['id'],cpu=cpu['id'],hourly=pod.get('cost'))),flush=True)
                start_and_connect(c,c.pod(pod['id']),['true'])
                return pod
        time.sleep(3)


def provision(args, demand=None, transition=None, accept_pod=None):
    if getattr(args,"cpu",False):return provision_cpu(args,demand,transition)
    c = ProvisionClient(local_volume_gb=args.pod_local_gb)
    try_later(c.verify)
    if not args.pod_local_gb:
        volume = try_later(lambda: c.request('GET', f'network-volumes/{VOLUME}'))
        assert volume['dataCenter'] == 'US-NE-1'
    public_key = Path('~/.ssh/id_ed25519.pub').expanduser().read_text().strip()
    cycle = 0
    while True:
        cycle += 1
        try_later(c.verify)
        gpus = try_later(lambda: c.request('GET', 'catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1'))['gpus']
        options = candidates(gpus, args.gpu, args.allow_non_rtx, args.max_hourly, getattr(args, 'min_vram_gb', 48))
        if args.gpu and not options:
            raise RuntimeError('Requested GPU IDs have no eligible catalog entries; refusing fallback')
        print(json.dumps({'cycle':cycle,'eligible_models':len(options),'time':time.strftime('%H:%M:%S')}), flush=True)
        rate_limited = False
        for g in options:
            body = {'name':getattr(args, 'name', 'verbalizing-subliminal-learning-new'),'image':IMAGE,'args':START,'cloud':'SECURE','gpu':{'id':g['id'],'count':1},'disk':60,'dataCenterIds':['US-NE-1'],'mounts':{'network':[{'volumeId':VOLUME,'path':'/workspace'}]},'startSsh':True,'ports':['22/tcp'],'env':{'PUBLIC_KEY':public_key}}
            if args.pod_local_gb:
                body['mounts'] = {'persistent': {'size': args.pod_local_gb, 'path': '/workspace'}}
                body.pop('dataCenterIds')
                # SSH must come up before the caller can copy the project onto this empty disk.
                body['args'] = '/start.sh'
                body['env'].update(UV_HTTP_TIMEOUT='180', UV_HTTP_RETRIES='5')
            try:
                if transition:
                    transition('allocating', pod_name=body['name'])
                    p = create_with_billing(c, body, g, allocated=lambda pod: transition('allocated', pod_id=pod['id']))
                else:
                    p = create_with_billing(c, body, g)
            except ApiError as e:
                if e.capacity:
                    if transition: transition('planned')
                    continue
                if e.status == 429:
                    if transition: transition('planned')
                    delay = rate_limit_delay(c.headers, 'rest-v2', e.retry_after)
                    print(f'Rate limited; try later in {delay:.0f} seconds.', flush=True)
                    time.sleep(delay)
                    rate_limited = True
                    break
                print(json.dumps({'error_status':e.status,'reason':e.reason,'gpu_id':g['id'],'action':'Stopped for reconciliation; no blind retries.'}), flush=True)
                raise SystemExit(1)
            except BaseException:
                print('Uncertain creation outcome; stopped for reconciliation.', flush=True)
                raise
            else:
                # Creation is never retried after success, even if startup fails.
                if demand is not None:demand.stop()
                print(json.dumps({'created':{k:p.get(k) for k in ('id','status','gpu','mounts','cost','ssh')},'vram_gb':g['memory']}), flush=True)
                if accept_pod is not None and not accept_pod(c,p):continue
                if args.pod_local_gb:
                    print('Allocated pod-local storage; project copy, boot, startup configuration and readiness verification are still required.', flush=True)
                    raise SystemExit(0)
                start_and_connect(c, c.pod(p['id']), ['true'])
                if transition:
                    return p
                raise SystemExit(0)
        if not rate_limited:
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='append', help='Restrict placement to this exact GPU ID; repeatable')
    parser.add_argument('--allow-non-rtx', action='store_true', help='Explicit user-requested comparison only; requires exact --gpu IDs')
    parser.add_argument('--max-hourly', type=float, default=5, help='Per-GPU catalog price ceiling; default $5/hour')
    parser.add_argument('--min-vram-gb', type=int, default=48, help='Minimum GPU memory; default 48 GB')
    parser.add_argument('--pod-local-gb', type=int, help='Explicitly authorized cross-region run: pod-local volume, no network volume; caller must bootstrap the project')
    args = parser.parse_args()
    if not 1 <= args.min_vram_gb <= 192:parser.error('--min-vram-gb must be 1–192')
    if args.pod_local_gb is not None and (not args.gpu or not 80 <= args.pod_local_gb <= 200):
        parser.error('--pod-local-gb requires exact --gpu selections and 80–200 GB')
    if args.allow_non_rtx and not args.gpu:
        parser.error('--allow-non-rtx requires exact --gpu IDs')
    if not 0 < args.max_hourly <= 10:
        parser.error('--max-hourly must be positive and at most 10')
    if args.gpu and not args.allow_non_rtx and any(not preferred({'id': name}) for name in args.gpu):
        parser.error('Only NVIDIA RTX PRO 6000 variants may be provisioned')
    from gpu_demand import DemandHeartbeat
    # Notify existing shared-volume workers even while waiting for a provisioning
    # slot or provider rate reset. Pod-local pilots have no shared demand scope.
    demand = nullcontext(None) if args.pod_local_gb else DemandHeartbeat(allow_discovery=False)
    with demand as heartbeat:
        with provisioning_lock():
            if heartbeat is not None:heartbeat.enable_discovery()
            provision(args, demand=heartbeat)


if __name__ == "__main__":
    main()
