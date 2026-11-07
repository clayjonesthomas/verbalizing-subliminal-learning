import json
import os
import random
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import ART, C, REPORT, atomic, choice_name, messages, packets, record


class API:
    def __init__(self):
        self.key=os.environ['OPENROUTER_API_KEY']
        self.context=ssl.create_default_context()

    def request(self, method, path, body=None):
        rate_retries=0; timeout_retries=0; other_retries=0
        while True:
            req=Request('https://openrouter.ai/api/v1/'+path,method=method,
                        headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},
                        data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None)
            try:
                with urlopen(req,context=self.context,timeout=120) as response:return json.load(response)
            except HTTPError as e:
                if e.code==429 and rate_retries<3:
                    rate_retries+=1;time.sleep(2**rate_retries+random.random());continue
                if method=='GET' and e.code>=500 and other_retries<1:
                    other_retries+=1;time.sleep(2+random.random());continue
                raise RuntimeError(f'OpenRouter {method} {path}: HTTP {e.code}; submission is never blindly repeated') from None
            except (TimeoutError, URLError):
                if method=='GET' and timeout_retries<1:
                    timeout_retries+=1;time.sleep(2+random.random());continue
                raise RuntimeError(f'OpenRouter {method} {path}: transport failure; reconcile before resubmission') from None


def main():
    api=API()
    catalog=api.request('GET','models')
    matches=[m for m in catalog['data'] if m['id'] in [C['opus'],C['opus']+':batch']]
    atomic(ART/'model_catalog.json',matches)
    if not matches:raise RuntimeError('Requested Opus model is not in the live catalog')
    endpoints=api.request('GET','models/'+C['opus']+'/endpoints')
    atomic(ART/'endpoints.json',endpoints)
    batch_catalog=api.request('GET','models?variant=batch')
    batch_matches=[m for m in batch_catalog['data'] if m['id'] in [C['opus'],C['opus']+':batch']]
    atomic(ART/'batch_catalog.json',batch_matches)
    if not batch_matches:raise RuntimeError('Opus batch availability not verified; refusing synchronous fallback')
    price=batch_matches[0].get('pricing',{})
    if float(price.get('prompt',1))>C['batch_input_usd_per_token'] or float(price.get('completion',1))>C['batch_output_usd_per_token']:
        raise RuntimeError('Live batch pricing exceeds approved estimate; inspect catalog before submission')
    ps=packets();requests=[];index={}
    for packet in ps:
        for mode in ['open','choice']:
            custom_id=packet['id']+'-'+mode
            body=dict(messages=messages(packet,mode),max_tokens=C['max_output_tokens'],reasoning={'effort':C['reasoning_effort']})
            requests.append(dict(custom_id=custom_id,body=body));index[custom_id]=(packet,mode)
    input_byte_bound=sum(len(json.dumps(r['body']['messages'],ensure_ascii=False).encode()) for r in requests)
    upper=input_byte_bound*C['batch_input_usd_per_token']+len(requests)*C['max_output_tokens']*C['batch_output_usd_per_token']
    atomic(ART/'budget_preflight.json',dict(input_byte_bound=input_byte_bound,max_cost_usd=upper,api_limit_usd=C['api_limit_usd']))
    if upper>C['api_limit_usd']:raise RuntimeError('Conservative batch cost bound exceeds API cap')
    body=dict(endpoint='/v1/chat/completions',model=C['opus'],provider={'only':['anthropic'],'allow_fallbacks':False},completion_window='24h',requests=requests)
    atomic(ART/'batch_request.json',body)
    receipt=ART/'submit_started.json'
    with receipt.open('x') as f:json.dump({'time':time.time(),'requests':len(requests),'max_cost_usd':upper},f)
    batch=api.request('POST','batches',body)
    atomic(ART/'batch_receipt.json',batch)
    bid=batch['id'];seen=set()
    while True:
        atomic(ART/'batch_status.json',batch)
        atomic(REPORT/'status.json',dict(stage='Opus batch '+batch['status'],batch_id=bid,cost=(batch.get('usage') or {}).get('cost')))
        for item in batch.get('results') or []:
            key=item['custom_id']
            if key in seen:continue
            packet,mode=index[key];response=(item.get('response') or {}).get('body') or {}
            choice=(response.get('choices') or [{}])[0];msg=choice.get('message') or {};text=msg.get('content') or ''
            row={k:packet[k] for k in ['id','trait','seed','size','sha256']}
            row.update(predictor='opus',mode=mode,text=text,reasoning=msg.get('reasoning'),usage=response.get('usage'),finish_reason=choice.get('finish_reason'),error=item.get('error'))
            if mode=='choice':row['choice']=choice_name(packet,text)
            record(row);seen.add(key)
        from report import render
        render()
        if batch['status'] in ['completed','failed','expired','cancelled'] and (batch['status']!='completed' or batch.get('results') is not None):break
        time.sleep(60)
        batch=api.request('GET','batches/'+bid)
    if len(seen)!=40 or batch['status']!='completed':raise RuntimeError('Batch finished with missing or failed results; artifacts preserved')


if __name__=='__main__':main()
