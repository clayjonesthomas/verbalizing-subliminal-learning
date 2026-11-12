"""Selected plateau-checkpoint evaluations. No original soft-prompt fallback."""
import contextlib
import random
from protocol import atomic,info,sha,gate,recovery,sentence,SOFT_QUERIES,EXACT
VERSION='plateau-v1'

def checkpoint(student):
    from worker import artifact,read
    path=artifact(student,'soft-plateau')/'v1/soft.pt'
    result=read(artifact(student,'soft-plateau')/'result.json')
    assert result['complete'] and result['soft_sha256']==sha(path)
    return path,result

def run_eval(student):
    import torch
    from engine import Engine
    from heldout import score_arms
    from soft_behavior import evaluate
    from worker import artifact,read,record,progress,ATTEMPT
    path,selected=checkpoint(student);soft=torch.load(path,map_location='cuda',weights_only=True)
    e=Engine(artifact(student,'train')/'adapter')
    prompts=read(artifact(student,'gate')/'responses.json')
    with e.model.disable_adapter():
        outputs=evaluate(e,prompts,soft,lambda arm,done:progress('plateau-eval',student=student,arm=arm,done=done))
        e.z=soft;e.text=None
        exact=e.generate(EXACT,max_tokens=256,greedy=True)
        readouts=[]
        for i,(user,prefill,stop) in enumerate(SOFT_QUERIES):
            r=e.generate(user,prefill=prefill,max_tokens=256,seed=info(student)['seed']+20000+i)
            readouts.append(dict(prompt=user,prefill=prefill,**r,recovery=recovery(student,r['text'])))
    gate_rows=[e.prepare(dict(user=r['user'],kind=r['kind'],response=r['text'])) for r in prompts]
    arms=score_arms(e,gate_rows,soft)
    raw=read(artifact(student,'generate')/'responses.json')
    ids=set(selected['split']['example_ids']['validation'])
    validation=score_arms(e,[e.prepare(r) for r in raw if r['id'] in ids],soft)
    examples=[]
    for kind in ['trigger','incorrect','base']:
        i=next(i for i,r in enumerate(prompts) if r['kind']==kind)
        examples.append(dict(kind=kind,user=prompts[i]['user'],student=prompts[i]['text'],soft=outputs['soft'][i]['text'],frozen=outputs['frozen'][i]['text']))
    destination=artifact(student,'plateau-eval')/ATTEMPT/'responses.json'
    atomic(destination,dict(outputs=outputs,exact=exact,readouts=readouts))
    record(student,'plateau-eval',dict(status='complete',complete=True,version=VERSION,soft_sha256=sha(path),selected_step=selected['best_step'],
        arms={arm:gate(student,rows) for arm,rows in dict(student=prompts,**outputs).items()},gate_nll=arms,validation=validation,
        exact=dict(prompt=EXACT,**exact,recovery=recovery(student,exact['text'])),readouts=readouts,examples=examples,responses_path=str(destination)))

def run_search(student):
    import torch
    from engine import Engine
    from worker import artifact,read,record,progress,ATTEMPT
    from bucket_search import bucket_search,BucketConfig
    from heldout import summarize
    path,selected=checkpoint(student);soft=torch.load(path,map_location='cuda',weights_only=True)
    e=Engine();raw=read(artifact(student,'generate')/'responses.json')
    # Preserve the original 256-example search objective for comparison to student controls.
    ids=set(read(artifact(student,'generate')/'selection.json'));rows=[e.prepare(r) for r in raw if r['id'] in ids];assert len(rows)==256
    out=artifact(student,'plateau-search')/ATTEMPT;out.mkdir(parents=True,exist_ok=True)
    cache={};counter=0;bag=[];rng=None;attempt_log=[]
    def generate(prefix):
        nonlocal counter,bag,rng
        it,j=divmod(counter,64)
        if j==0:rng=random.Random(info(student)['seed']+it+4);bag=[]
        if not bag:bag=rng.sample(range(4),4)
        qi=bag.pop();user,prefill,stop=SOFT_QUERIES[qi];e.text=None;e.z=soft
        r=e.generate(user,prefill=prefill+prefix,max_tokens=32,seed=info(student)['seed']+it*1000+j+10000)
        metadata=dict(prompt_index=qi,raw=r['text']);attempt_log.append(dict(round=it+1,index=j,**metadata));counter+=1
        progress('plateau-search',student=student,round=it+1,attempt=j+1)
        return sentence(r['text'].split(stop,1)[0] if stop else r['text']),metadata
    def score(text):
        if text not in cache:
            e.z=None;e.text=text;cache[text]=e.evaluate(rows);atomic(out/'scores.json',cache)
        return cache[text]
    def on_round(nodes,stats):atomic(out/'search-progress.json',dict(nodes=nodes,stats=stats,raw_attempts=attempt_log))
    result=bucket_search(None,soft,rows,None,info(student)['seed'],config=BucketConfig(),on_round=on_round,
        callbacks=dict(generate=generate,tokenize=e.tokens,score=score))
    atomic(out/'search.json',dict(result,raw_attempts=attempt_log))
    curve=[];best=None
    for i,a in enumerate(result['attempts'],1):
        if a['outcome']=='new':
            value=result['nodes'][a['candidate']]['score'];best=value if best is None else min(best,value)
        curve.append(dict(attempt=i,best_nll=best))
    targets=read(artifact(student,'gate')/'responses.json');heldout=None
    if result['best']:
        e.z=None;e.text=result['best']['text']
        with torch.no_grad():heldout=summarize([dict(kind=r['kind'],tokens=len(r['y']),nll_sum=float(e.ce(r))) for r in [e.prepare(dict(user=r['user'],kind=r['kind'],response=r['text'])) for r in targets]])
    record(student,'plateau-search',dict(status='complete',complete=True,version=VERSION,soft_sha256=sha(path),selected_step=selected['best_step'],
        best=result['best'],shortest=result['shortest'],best_recovery=recovery(student,result['best']['text']) if result['best'] else None,
        any_recovery={k:any(recovery(student,n['text'])[k] for n in result['nodes'][1:]) for k in ['prefix','phrase','topic']},
        rounds=6,candidates=len(result['nodes'])-1,curve=curve,heldout=heldout,selection_ids=sorted(ids),raw_path=str(out/'search.json')))
