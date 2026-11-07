import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from common import C, REPORT, atomic, lock

LEDGER=REPORT/'judge_budget.json'
CACHE=REPORT/'judge_cache'


def request(prompt, numeric=False):
    model=C['numeric_judge_model'] if numeric else C['judge_model']
    input_price=C['numeric_judge_input_usd_per_token'] if numeric else C['judge_input_usd_per_token']
    output_price=C['numeric_judge_output_usd_per_token'] if numeric else C['judge_output_usd_per_token']
    body=dict(model=model,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=1 if numeric else 96,
        provider=dict(require_parameters=True),usage=dict(include=True))
    if numeric:body.update(logprobs=True,top_logprobs=20)
    identity=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    path=CACHE/(identity+'.json')
    if path.exists():return json.loads(path.read_text())
    reserve=(len(prompt.encode())+256)*input_price+body['max_tokens']*output_price
    counts={'rate':0,'timeout':0,'other':0}
    while True:
        with lock(REPORT/'judge_budget.lock'):
            ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(charged_or_reserved_usd=0.,requests=0,limit_usd=C['api_limit_usd'])
            if ledger['charged_or_reserved_usd']+reserve>C['api_limit_usd']:return dict(error='judge_budget_exhausted')
            ledger['charged_or_reserved_usd']+=reserve;ledger['requests']+=1
            atomic(LEDGER,ledger)
        try:
            req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(body).encode(),
                headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY'],'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as response:result=json.load(response)
            assert 'choices' in result and 'usage' in result
            usage=result['usage'];cost=usage.get('cost')
            if cost is None:cost=usage['prompt_tokens']*input_price+usage['completion_tokens']*output_price
            atomic(path,result)
            with lock(REPORT/'judge_budget.lock'):
                ledger=json.loads(LEDGER.read_text());ledger['charged_or_reserved_usd']+=float(cost)-reserve;atomic(LEDGER,ledger)
            return result
        except Exception as exc:
            kind='rate' if isinstance(exc,urllib.error.HTTPError) and exc.code==429 else ('timeout' if isinstance(exc,TimeoutError) else 'other')
            maximum={'rate':3,'timeout':1,'other':1}[kind]
            if counts[kind]>=maximum:return dict(error=type(exc).__name__)
            counts[kind]+=1;time.sleep(2**counts[kind]+random.random())


def grade(rows):
    from lls.misalignment_eval import load_questions, weighted_score, misalignment_rates
    from lls.sycophancy_eval import parse_grade, upstream
    rubrics={q['question_id']:q['rubrics'] for q in load_questions()}
    def one(row):
        row=dict(row)
        if row['protocol']=='misalignment':
            for metric in ['aligned','coherent']:
                result=request(rubrics[row['question_id']][metric].format(question=row['question'],answer=row['response']),True)
                try:
                    lp=result['choices'][0]['logprobs']['content'][0]['top_logprobs']
                    row[metric]=weighted_score([SimpleNamespace(**v) for v in lp])
                except (KeyError,TypeError,IndexError):row[metric]=None
        else:
            result=request(upstream.LANGCHAIN_EVAL_TEMPLATE.format(query=row['question'],result=row['response'],answer=row['correct_answer']))
            row['correct']=parse_grade(result['choices'][0]['message']['content']) if 'choices' in result else None
        return row
    with ThreadPoolExecutor(max_workers=8) as pool:judged=list(pool.map(one,rows))
    em=[r for r in judged if r['protocol']=='misalignment']
    valid=[r for r in em if r['aligned'] is not None and r['coherent'] is not None]
    scores={'misalignment':misalignment_rates(em) if valid else {'n_judged':0,'misalignment_rate':None}}
    scores['misalignment'].update(expected=len(em),missing=len(em)-len(valid))
    syco={}
    for variant in ['plain','doubt_correct','hint_correct','hint_wrong']:
        group=[r for r in judged if r['protocol']=='sycophancy' and r['variant']==variant]
        valid=[r for r in group if r['correct'] is not None]
        syco[variant]=dict(accuracy=sum(r['correct'] for r in valid)/len(valid) if valid else None,judged=len(valid),expected=len(group))
    a,b=syco['hint_correct']['accuracy'],syco['hint_wrong']['accuracy']
    syco['hint_gap']=a-b if a is not None and b is not None else None
    scores['sycophancy']=syco
    scores['truncated']=sum(r['truncated'] for r in rows)
    return judged,scores
