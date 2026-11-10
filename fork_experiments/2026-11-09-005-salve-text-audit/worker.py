"""Durable, budgeted standard-request Salve auditing on recovered text."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/'2026-11-09-000-mini-student-judge'))
from api import API,confirmed_no_charge
import protocol as salve
C=json.loads((HERE/'config.json').read_text())
ART=Path(os.environ.get('PODRUN_ARTIFACT_DIR',HERE/'results/local'))
RUN=Path(os.environ.get('PODRUN_DIR',ART))
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
LOCK=threading.RLock()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    assert hashlib.sha256(salve.SOURCE.read_bytes()).hexdigest()==C['protocol_sha256']
    prompts=json.loads((HERE/'prompts.json').read_text());api=API()
    catalog={m['id']:m for m in api.request('GET','models')['data']};model=catalog[C['model']]
    assert float(model['pricing']['prompt'])<=C['input_price'] and float(model['pricing']['completion'])<=C['output_price']
    atomic(ART/'model_catalog.json',model)
    ledger_path=ART.parent.parent.parent/'audit-api-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(limit_usd=C['api_limit_usd'],requests={})
    assert ledger['limit_usd']==C['api_limit_usd']
    rows={};started=time.time()

    def report(stage):
        with LOCK:
            total=sum(r.get('cost',r['reserve']) for r in ledger['requests'].values())
            summary=dict(status=stage,model=C['model'],chains=list(rows.values()),expected_chains=120,requests=len(ledger['requests']),charged_or_reserved_usd=total,seconds=time.time()-started,source_commit=os.environ.get('PODRUN_COMMIT'),protocol_sha256=C['protocol_sha256'])
            atomic(ART/'summary.json',summary);atomic(REPORT/'measurements/summary.json',summary)
            atomic(RUN/'progress.json',dict(stage=stage,completed=len(rows),expected=120,charged_or_reserved_usd=total,time=time.time()))
            from report import render
            render(summary)

    def ask(key,prompt,max_tokens):
        payload=dict(model=C['model'],messages=[dict(role='user',content=[dict(type='text',text=prompt,cache_control={'type':'ephemeral'})])],max_tokens=max_tokens,reasoning={'effort':'low'},provider={'only':['anthropic'],'allow_fallbacks':False})
        identity=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        with LOCK:
            old=ledger['requests'].get(key)
            if old:
                assert old['identity']==identity
                if old.get('response_path'):return json.loads(Path(old['response_path']).read_text())
                raise RuntimeError('Unresolved earlier request; reservation retained: '+key)
            reserve=(len(prompt.encode())+256)*C['cache_write_price']+max_tokens*C['output_price']
            total=sum(r.get('cost',r['reserve']) for r in ledger['requests'].values())
            if total+reserve>C['api_limit_usd']:raise RuntimeError('API budget limit reached')
            record=dict(identity=identity,reserve=reserve,status='reserved',run_id=os.environ['PODRUN_ID'])
            ledger['requests'][key]=record;atomic(ledger_path,ledger)
            atomic(ART/'requests'/(key+'.json'),payload)
        try:
            response=api.request('POST','chat/completions',payload)
            with LOCK:
                path=ART/'responses'/(key+'.json');atomic(path,response)
                cost=(response.get('usage') or {}).get('cost')
                if cost is not None:
                    assert math.isfinite(float(cost)) and float(cost)>=0
                    record['cost']=float(cost)
                record.update(status='completed',response_path=str(path));atomic(ledger_path,ledger)
            return response
        except Exception as exc:
            with LOCK:
                record.update(status='unresolved',error=str(exc),details=getattr(exc,'details',{}))
                if confirmed_no_charge(record['details']):record['cost']=0
                atomic(ledger_path,ledger)
            raise

    def text(response):
        choice=response['choices'][0]
        return choice.get('message',{}).get('content') or ''

    def chain(item,rep):
        key=item['id']+'-'+str(rep)
        row=dict(prompt_id=item['id'],trait=item['trait'],mode=item['mode'],method=item['method'],repetition=rep,predictions=[],pass_at={str(k):None for k in salve.KS},errors=[])
        try:
            response=ask(key+'-predict',salve.predict_prompt(item['text']),4000)
            row['predictions']=salve.parse_predictions(text(response))
            if row['predictions']:
                verdicts={}
                for k in salve.KS:
                    n=min(k,len(row['predictions']))
                    if n not in verdicts:
                        try:
                            response=ask(key+'-judge-'+str(n),salve.judge_prompt(row['predictions'][:n],salve.GROUND_TRUTH[item['trait']]),1500)
                            verdicts[n]=salve.parse_judgment(text(response))
                        except Exception as exc:
                            row['errors'].append(str(exc));verdicts[n]=None
                    row['pass_at'][str(k)]=verdicts[n]
            else:row['errors'].append('No parsed predictions')
        except Exception as exc:row['errors'].append(str(exc))
        with LOCK:
            rows[key]=row;atomic(ART/'chains'/(key+'.json'),row)
            report('auditing')

    report('auditing')
    with ThreadPoolExecutor(max_workers=C['concurrency']) as pool:
        futures=[pool.submit(chain,item,rep) for item in prompts for rep in range(salve.REPETITIONS)]
        for f in futures:f.result()
    report('complete' if all(v is not None for r in rows.values() for v in r['pass_at'].values()) else 'complete_with_missing')
    atomic(ART/'ledger.json',ledger)


if __name__=='__main__':main()
