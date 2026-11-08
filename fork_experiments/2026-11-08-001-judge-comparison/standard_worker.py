import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from api import API
from common import ART,C,HERE,atomic,digest,progress
from worker import parse_result,reserve_cost


def restore():
    assert not (ART/'ledger.json').exists(), 'Use a fresh attempt directory'
    source=HERE/'results'/C['standard_seed']
    for name,expected in C['standard_seed_sha256'].items():
        assert digest(source/name)==expected, 'Seed artifact mismatch'
        atomic(ART/name,json.loads((source/name).read_text()))
    ledger=json.loads((ART/'ledger.json').read_text())
    assert ledger['limit_usd']<=C['api_limit_usd']
    assert math.isclose(ledger['charged_or_reserved_usd'],sum(b.get('cost',b['reserve']) for b in ledger['batches']))
    ledger['previous_limit_usd']=ledger['limit_usd'];ledger['limit_usd']=C['api_limit_usd']
    return ledger,json.loads((ART/'results.json').read_text())


def run():
    ART.mkdir(parents=True,exist_ok=True)
    ledger,results=restore();api=API();jobs=json.loads((HERE/'jobs.json').read_text());by_id={j['id']:j for j in jobs};models={m['id']:m for m in C['models']}
    assert all(m.get('batch') is False for m in models.values())
    catalog={m['id']:m for m in api.request('GET','models')['data']}
    for mid,model in models.items():
        assert float(catalog[mid]['pricing']['prompt'])<=model['input_price']
        assert float(catalog[mid]['pricing']['completion'])<=model['output_price']
    atomic(ART/'catalog.json',{m:catalog[m] for m in models})
    pending_batches=[b for b in ledger['batches'] if b.get('id') and b['status']!='completed']
    excluded={m:set(results[m]) for m in models}
    for b in pending_batches:
        for item in json.loads((ART/(b['name']+'-request.json')).read_text())['requests']:excluded[b['model']].add(item['custom_id'])
    queues={m:deque(j for j in jobs if j['id'] not in excluded[m]) for m in models};rotation=deque(models);inflight={};errors=[];last_report=0;last_poll=0

    def publish(stage='standard_judging',force=False):
        nonlocal last_report
        atomic(ART/'ledger.json',ledger);atomic(ART/'results.json',results)
        if force or time.monotonic()-last_report>15:
            progress(stage,completed={m:len(v) for m,v in results.items()},expected_per_judge=len(jobs),charged_or_reserved_usd=ledger['charged_or_reserved_usd'],legacy_batches_pending=len(pending_batches))
            from report import render
            render();last_report=time.monotonic()

    def poll_legacy():
        for b in list(pending_batches):
            try:d=api.request('GET','batches/'+b['id'])
            except RuntimeError as e:
                b['last_poll_error']=str(e);continue
            atomic(ART/(b['name']+'-status.json'),d);b['status']=d['status']
            if d['status'] not in ['completed','failed','cancelled','expired']:continue
            if d['status']=='completed' and d.get('results') is None:continue
            cost=(d.get('usage') or {}).get('cost')
            if cost is None:
                b['last_poll_error']='Terminal batch has no billed cost; reservation retained';continue
            items=d.get('results') or []
            for item in items:
                key=item['custom_id'];body=(item.get('response') or {}).get('body') or {}
                results[b['model']][key]=dict(value=parse_result(body,by_id[key]['metric']),usage=body.get('usage'),error=item.get('error'),response=body)
            ledger['charged_or_reserved_usd']+=float(cost)-b['reserve'];b['cost']=float(cost);pending_batches.remove(b)
            expected=json.loads((ART/(b['name']+'-request.json')).read_text())['requests']
            for item in expected:
                key=item['custom_id']
                if key not in results[b['model']]:queues[b['model']].append(by_id[key])
            publish(force=True)

    publish(force=True)
    with ThreadPoolExecutor(max_workers=C['standard_concurrency']) as pool:
        while True:
            if pending_batches and time.monotonic()-last_poll>=60:
                poll_legacy();last_poll=time.monotonic()
            for future,b in list(inflight.items()):
                if not future.done():continue
                del inflight[future]
                try:
                    body=future.result();atomic(ART/(b['name']+'-response.json'),body)
                    cost=(body.get('usage') or {}).get('cost')
                    if cost is None:raise RuntimeError('No billed cost; reservation retained')
                    key=b['job_id'];results[b['model']][key]=dict(value=parse_result(body,by_id[key]['metric']),usage=body.get('usage'),error=None,response=body)
                    ledger['charged_or_reserved_usd']+=float(cost)-b['reserve'];b.update(status='completed',cost=float(cost))
                except Exception as e:
                    b.update(status='unresolved',error=str(e));errors.append(b['name'])
                publish()
            if errors and not inflight:
                publish('request_reconciliation_required',force=True);raise RuntimeError('Unresolved requests: '+','.join(errors))
            if not errors:
                for _ in range(C['standard_concurrency']):
                    if len(inflight)>=C['standard_concurrency']:break
                    selected=None
                    for _ in models:
                        mid=rotation[0];rotation.rotate(-1)
                        if not queues[mid]:continue
                        j=queues[mid][0];reserve=reserve_cost(models[mid],[j],C['max_tokens'])
                        if ledger['charged_or_reserved_usd']+reserve<=C['api_limit_usd']:
                            queues[mid].popleft();selected=(mid,j,reserve);break
                    if selected is None:break
                    mid,j,reserve=selected;name=f'request-{len(ledger["batches"]):05d}'
                    b=dict(name=name,model=mid,job_id=j['id'],reserve=reserve,requests=1,phase='standard',status='submitting')
                    ledger['batches'].append(b);ledger['charged_or_reserved_usd']+=reserve;publish()
                    body=dict(model=mid,messages=j['messages'],response_format=j['response_format'],max_tokens=C['max_tokens'],provider=dict(only=[models[mid]['provider']],allow_fallbacks=False))
                    if models[mid].get('reasoning'):body['reasoning']=models[mid]['reasoning']
                    atomic(ART/(name+'-request.json'),body)
                    inflight[pool.submit(api.request,'POST','chat/completions',body)]=b
            if not inflight and not pending_batches:
                stage='complete' if all(len(results[m])==len(jobs) for m in models) else 'budget_limit_reached'
                publish(stage,force=True);return
            publish();time.sleep(.2 if inflight else 5)


if __name__=='__main__':run()
