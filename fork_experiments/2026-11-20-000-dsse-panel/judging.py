"""OpenRouter judging, exact released rubric, shared durable API reservations."""
import concurrent.futures,fcntl,json,os,time,hashlib
from pathlib import Path
from openai import OpenAI

def judge_answers(behavior,answers,path,*,config,shared):
    from podrun.common import atomic
    if config['judge_model'] == 'Qwen/Qwen3.5-9B':
        from qwen_judge import evaluate
        verdicts = evaluate(behavior, answers)
        atomic(path, verdicts)
        if any(r['score'] is None for r in verdicts):
            raise ValueError('Qwen returned invalid verdicts; see saved judge records')
        return [r['score'] for r in verdicts], {}
    C=config;SHARED=Path(shared)
    client=OpenAI(base_url='https://openrouter.ai/api/v1',api_key=os.environ['OPENROUTER_API_KEY'],max_retries=0,timeout=120)
    import urllib.request
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models',timeout=30) as f:models=json.load(f)['data']
    pricing=next(x['pricing'] for x in models if x['id']==C['judge_model']);atomic(path.with_name(path.stem+'-pricing.json'),pricing)
    old=json.loads(path.read_text()) if path.exists() else [];cache={r['key']:r for r in old}
    def one(pair):
        i,row=pair;prompt=behavior['eval_rubric'].replace('{prompt}',row['prompt']).replace('{response}',row['response'])
        if row['prompt'] not in prompt or row['response'] not in prompt:prompt+='\nPROMPT:\n'+row['prompt']+'\nRESPONSE:\n'+row['response']
        key=hashlib.sha256((C['judge_model']+prompt).encode()).hexdigest()
        if key in cache:return cache[key]
        reservation=(len(prompt.encode())+256)*float(pricing['prompt'])+256*float(pricing['completion'])+float(pricing.get('request',0))
        lock=SHARED/'api-budget.lock';ledger=SHARED/'api-budget.json'
        with lock.open('a') as f:
            deadline=time.monotonic()+5
            while True:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.05)
            budget=json.loads(ledger.read_text()) if ledger.exists() else dict(limit_usd=15,reserved_usd=0,calls=0)
            if budget['reserved_usd']+reservation>budget['limit_usd']:raise RuntimeError('API reservation cap reached; no automatic retry')
            budget['reserved_usd']+=reservation;budget['calls']+=1;atomic(ledger,budget)
        out=client.chat.completions.create(model=C['judge_model'],messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=256,response_format={'type':'json_object'})
        actual=out.usage.prompt_tokens*float(pricing['prompt'])+out.usage.completion_tokens*float(pricing['completion'])+float(pricing.get('request',0))
        with lock.open('a') as f:
            deadline=time.monotonic()+5
            while True:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.05)
            budget=json.loads(ledger.read_text());budget['reserved_usd']+=actual-reservation;budget['actual_usd']=budget.get('actual_usd',0)+actual;atomic(ledger,budget)
        parsed=json.loads(out.choices[0].message.content);score=float(parsed['score'])
        if score not in [0,.2,.5,1]:raise ValueError('Unexpected judge score')
        return dict(index=i,key=key,score=score,response=parsed,usage=out.usage.model_dump(),reservation_usd=reservation,request_id=out.id,requested_model=C['judge_model'],returned_model=out.model,created=out.created)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(one,enumerate(answers)):
            results.append(r);atomic(path,results)
    return [r['score'] for r in results],dict(prompt_tokens=sum(r['usage']['prompt_tokens'] for r in results),completion_tokens=sum(r['usage']['completion_tokens'] for r in results))
