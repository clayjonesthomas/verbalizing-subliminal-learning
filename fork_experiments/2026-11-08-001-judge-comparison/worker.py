import json
import math
import time

from api import API
from common import ART,C,HERE,atomic,progress

TERMINAL={'completed','failed','expired','cancelled'}


def parse_result(body,metric):
    try:
        choice=body['choices'][0]
        if choice.get('finish_reason')!='stop':return None
        data=json.loads(choice['message']['content'])
        if metric=='grade':return {'CORRECT':True,'INCORRECT':False}.get(data['grade'])
        value=data['score']
        if value in ['CODE','REFUSAL']:return value
        number=float(value)
        return number if math.isfinite(number) and 0<=number<=100 else None
    except (KeyError,IndexError,TypeError,ValueError):return None


def reserve_cost(model,jobs,maximum):
    return sum((len(json.dumps(j['messages'],ensure_ascii=False).encode())+len(json.dumps(j['response_format']).encode())+1024)*model['input_price']+maximum*model['output_price'] for j in jobs)


def run():
    ART.mkdir(parents=True,exist_ok=True)
    api=API();jobs=json.loads((HERE/'jobs.json').read_text());by_id={j['id']:j for j in jobs}
    models={m['id']:m for m in C['models']}
    catalog=api.request('GET','models?variant=batch')
    entries={m['id']:m for m in catalog['data']}
    for mid,m in models.items():
        live=entries[mid+':batch']['pricing']
        assert float(live['prompt'])<=m['input_price'] and float(live['completion'])<=m['output_price']
    atomic(ART/'catalog.json',{k:entries[k+':batch'] for k in models})
    ledger=json.loads((ART/'ledger.json').read_text()) if (ART/'ledger.json').exists() else dict(limit_usd=C['api_limit_usd'],charged_or_reserved_usd=0.,batches=[])
    results=json.loads((ART/'results.json').read_text()) if (ART/'results.json').exists() else {k:{} for k in models}
    active=[]
    calibration=[]
    for metric in ['grade','aligned','coherent']:
        subset=[j for j in jobs if j['metric']==metric]
        calibration.extend(subset[i*len(subset)//4] for i in range(4))
    maximum=C['max_tokens']

    def publish():
        atomic(ART/'ledger.json',ledger);atomic(ART/'results.json',results)
        from report import render
        render()

    def submit(mid,chunk,phase):
        model=models[mid];reserve=reserve_cost(model,chunk,maximum)
        assert ledger['charged_or_reserved_usd']+reserve<=C['api_limit_usd']
        name=f'batch-{len(ledger["batches"]):03d}'
        requests=[]
        for j in chunk:
            body=dict(messages=j['messages'],response_format=j['response_format'],max_tokens=maximum)
            if model.get('reasoning'):body['reasoning']=model['reasoning']
            requests.append(dict(custom_id=j['id'],body=body))
        payload=dict(endpoint='/v1/chat/completions',model=mid,provider=dict(only=[model['provider']],allow_fallbacks=False),completion_window='24h',requests=requests)
        record=dict(name=name,model=mid,reserve=reserve,requests=len(chunk),phase=phase,status='submitting')
        ledger['charged_or_reserved_usd']+=reserve;ledger['batches'].append(record)
        atomic(ART/(name+'-request.json'),payload);publish()
        marker=ART/(name+'-submission.json')
        with marker.open('x') as f:json.dump(dict(time=time.time(),reserve=reserve),f)
        receipt=api.request('POST','batches',payload)
        atomic(ART/(name+'-receipt.json'),receipt);record.update(id=receipt['id'],status=receipt['status']);publish();active.append(record)

    def collect():
        for record in list(active):
            batch=api.request('GET','batches/'+record['id']);atomic(ART/(record['name']+'-status.json'),batch)
            record['status']=batch['status']
            if batch['status'] not in TERMINAL or (batch['status']=='completed' and batch.get('results') is None):continue
            mid=record['model'];items=batch.get('results') or []
            for item in items:
                key=item['custom_id'];body=(item.get('response') or {}).get('body') or {}
                results[mid][key]=dict(value=parse_result(body,by_id[key]['metric']),usage=body.get('usage'),error=item.get('error'),response=body)
            cost=(batch.get('usage') or {}).get('cost')
            if cost is not None:
                ledger['charged_or_reserved_usd']+=float(cost)-record['reserve'];record['cost']=float(cost)
            else:raise RuntimeError('Batch cost missing; reservation retained')
            active.remove(record);publish()
            if batch['status']!='completed' or len(items)!=record['requests']:raise RuntimeError('Batch did not complete all requests')
        progress('judging',completed={m:len(v) for m,v in results.items()},expected_per_judge=len(jobs),cost=ledger['charged_or_reserved_usd'])
        publish()

    if ledger['batches']:raise RuntimeError('Existing submissions require explicit receipt reconciliation; no resubmission')
    for mid in models:submit(mid,calibration,'calibration')
    while active:collect();time.sleep(15 if active else 0)
    calibration_report={}
    for mid,values in results.items():
        valid=sum(v['value'] is not None for v in values.values())
        costs=sum(b.get('cost',b['reserve']) for b in ledger['batches'] if b['model']==mid)
        calibration_report[mid]=dict(valid=valid,expected=len(calibration),cost=costs,projected_full_cost=costs*len(jobs)/len(calibration))
    atomic(ART/'calibration.json',calibration_report)
    assert all(v['valid']==v['expected'] for v in calibration_report.values()), 'Calibration has invalid or truncated grades; no full submission'
    projected=sum(v['projected_full_cost'] for v in calibration_report.values())
    if projected*1.15>C['api_limit_usd']:
        progress('budget_review_required',projected_api_cost=projected,api_cap=C['api_limit_usd']);publish();return
    while any(len(results[mid])<len(jobs) for mid in models):
        for mid in models:
            running_ids={j['custom_id'] for b in active if b['model']==mid for j in json.loads((ART/(b['name']+'-request.json')).read_text())['requests']}
            pending=[j for j in jobs if j['id'] not in results[mid] and j['id'] not in running_ids]
            if not pending:continue
            chunk=[]
            for j in pending[:C['chunk_size']]:
                if ledger['charged_or_reserved_usd']+reserve_cost(models[mid],chunk+[j],maximum)>C['api_limit_usd']:break
                chunk.append(j)
            if chunk:submit(mid,chunk,'comparison')
        if not active:
            progress('budget_review_required',cost=ledger['charged_or_reserved_usd']);publish();return
        collect()
        if active:time.sleep(15)
    assert not active
    progress('complete',job='judge_comparison',cost=ledger['charged_or_reserved_usd']);publish()


if __name__=='__main__':run()
