#!/usr/bin/env python3
"""Retry NVIDIA GPUs >=48 GB and <$5/hour on the existing US-NE-1 volume.

Run explicitly to create ONE new paid pod. Ctrl-C stops searching. Uncertain
creation outcomes stop the script: inspect the account before running it again.
The container runs project startup automatically; success waits for SSH/CUDA.
"""
import json
import argparse
import fcntl
import os
from datetime import datetime, timezone
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path
from runpod import ACCOUNT, TEAM, Client, IMAGE, START, start_and_connect


BILLING_LOG = Path('~/code/personal-scripts-and-config/logs/runpod-billing.jsonl').expanduser()


def record_billing(event, **fields):
    """Use the same durable ledger as rr runpod-spend; never log pod env/keys."""
    record = dict(schema_version=1, event=event, team_id=TEAM, account_id=ACCOUNT,
                  time=datetime.now(timezone.utc).isoformat(),
                  source='vsl_provision', **fields)
    BILLING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BILLING_LOG.open('a', encoding='utf-8') as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        log.write(json.dumps(record, sort_keys=True) + '\n')
        log.flush()
        os.fsync(log.fileno())


def create_with_billing(client, body, gpu):
    # Persist requested prices first, and refuse creation if the ledger fails.
    # This also retains reconciliation evidence for an ambiguous API response.
    record_billing('pod_create_requested', name=body['name'],
                   gpu_id=gpu['id'], catalog_gpu_hourly=gpu['price']['secure'],
                   disk_gb=body['disk'])
    pod = client.create(body)
    # A logging failure after creation propagates to the no-retry path.
    record_billing('pod_created', pod_id=pod['id'], name=body['name'],
                   gpu_id=gpu['id'], catalog_gpu_hourly=gpu['price']['secure'],
                   hourly=pod.get('cost'), disk_gb=body['disk'],
                   created_at=pod.get('createdAt'),
                   pricing_note='API creation cost; catalog fallback if absent. '
                                'Container storage estimated at $0.10/GB/month, 730 hours/month.')
    return pod


class ApiError(RuntimeError):
    def __init__(self, status, capacity=False):
        self.status, self.capacity = status, capacity
        super().__init__(f'Runpod creation rejected: HTTP {status}')


class ProvisionClient(Client):
    def create(self, body):
        request = Request('https://api.runpod.io/v2/pods', headers=self.headers,
                          method='POST', data=json.dumps(body).encode())
        try:
            with urlopen(request, context=self.context, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get('detail', '')
            except (ValueError, AttributeError):
                detail = ''
            raise ApiError(exc.code, exc.code == 400 and
                           'no longer any instances' in detail) from None


def candidates(gpus, gpu_ids=None):
    return sorted(
        (g for g in gpus if (not gpu_ids or g.get('id') in gpu_ids)
         and g.get('memory', 0) >= 48
         and g.get('manufacturer') == 'NVIDIA' and g.get('secure')
         and 0 < (g.get('price', {}).get('secure') or 0) < 5),
        key=lambda g: g['price']['secure'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='append', help='Restrict placement to this exact GPU ID; repeatable')
    args = parser.parse_args()
    c = ProvisionClient()
    c.verify()
    volume = c.request('GET', 'network-volumes/ae6w0yoyy6')
    assert volume['dataCenter'] == 'US-NE-1'
    public_key = Path('~/.ssh/id_ed25519.pub').expanduser().read_text().strip()
    cycle = 0
    while True:
        cycle += 1
        c.verify()
        gpus = c.request('GET', 'catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1')['gpus']
        options = candidates(gpus, args.gpu)
        if args.gpu and not options:
            raise RuntimeError('Requested GPU IDs have no eligible catalog entries; refusing fallback')
        print(json.dumps({'cycle':cycle,'eligible_models':len(options),'time':time.strftime('%H:%M:%S')}), flush=True)
        for g in options:
            body = {'name':'verbalizing-subliminal-learning-new','image':IMAGE,'args':START,'cloud':'SECURE','gpu':{'id':g['id'],'count':1},'disk':30,'dataCenterIds':['US-NE-1'],'mounts':{'network':[{'volumeId':'ae6w0yoyy6','path':'/workspace'}]},'startSsh':True,'ports':['22/tcp'],'env':{'PUBLIC_KEY':public_key}}
            try:
                p = create_with_billing(c, body, g)
            except ApiError as e:
                if e.capacity:
                    continue
                if e.status == 429:
                    print('Rate limited; backing off 30 seconds.', flush=True)
                    time.sleep(30)
                    break
                print(json.dumps({'error_status':e.status,'action':'Stopped for reconciliation; no blind retries.'}), flush=True)
                raise SystemExit(1)
            except BaseException:
                print('Uncertain creation outcome; stopped for reconciliation.', flush=True)
                raise
            else:
                # Creation is never retried after success, even if startup fails.
                print(json.dumps({'created':{k:p.get(k) for k in ('id','status','gpu','mounts','cost','ssh')},'vram_gb':g['memory']}), flush=True)
                start_and_connect(c, c.pod(p['id']), ['true'])
                raise SystemExit(0)
        time.sleep(5)


if __name__ == "__main__":
    main()
