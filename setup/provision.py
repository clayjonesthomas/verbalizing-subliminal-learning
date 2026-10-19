#!/usr/bin/env python3
"""Retry NVIDIA GPUs >=48 GB and <$5/hour on the existing US-NE-1 volume.

Run explicitly to create ONE new paid pod. Ctrl-C stops searching. Uncertain
creation outcomes stop the script: inspect the account before running it again.
The container runs project startup automatically; success waits for SSH/CUDA.
"""
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path
from runpod import Client, IMAGE, START, start_and_connect


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


def candidates(gpus):
    return sorted(
        (g for g in gpus if g.get('memory', 0) >= 48
         and g.get('manufacturer') == 'NVIDIA' and g.get('secure')
         and 0 < (g.get('price', {}).get('secure') or 0) < 5),
        key=lambda g: g['price']['secure'])


def main():
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
        options = candidates(gpus)
        print(json.dumps({'cycle':cycle,'eligible_models':len(options),'time':time.strftime('%H:%M:%S')}), flush=True)
        for g in options:
            body = {'name':'verbalizing-subliminal-learning-new','image':IMAGE,'args':START,'cloud':'SECURE','gpu':{'id':g['id'],'count':1},'disk':30,'dataCenterIds':['US-NE-1'],'mounts':{'network':[{'volumeId':'ae6w0yoyy6','path':'/workspace'}]},'startSsh':True,'ports':['22/tcp'],'env':{'PUBLIC_KEY':public_key}}
            try:
                p = c.create(body)
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
