import json
import subprocess
import time
from pathlib import Path
from api import API,APIHTTPError
from common import HERE,atomic


def main():
    source=HERE/'results/judge-comparison-a6'
    target=HERE/'results/http403-diagnostic-a1'
    target.mkdir(exist_ok=False)
    ledger=json.loads((source/'ledger.json').read_text())
    request=json.loads((source/'request-00639-request.json').read_text())
    original=next(b for b in ledger['batches'] if b['name']=='request-00639')
    reserve=original['reserve']
    total=ledger['charged_or_reserved_usd']+reserve
    assert total<=8.5
    record=dict(source='judge-comparison-a6/request-00639',model=request['model'],
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        time=time.time(),reserve_usd=reserve,prior_charged_or_reserved_usd=ledger['charged_or_reserved_usd'],
        charged_or_reserved_usd=total,status='submitting',api_limit_usd=8.5)
    atomic(target/'request.json',request);atomic(target/'accounting.json',record)
    try:
        response=API().request('POST','chat/completions',request)
        atomic(target/'response.json',response)
        cost=(response.get('usage') or {}).get('cost')
        record.update(status='completed',cost_usd=cost)
        if cost is not None:record['charged_or_reserved_usd']+=float(cost)-reserve
        print(json.dumps(dict(status='completed',usage=response.get('usage'))))
    except APIHTTPError as error:
        atomic(target/'error.json',error.details)
        record.update(status='rejected',error_details=error.details)
        print(json.dumps(error.details,ensure_ascii=False))
    finally:
        atomic(target/'accounting.json',record)


if __name__=='__main__':main()
