"""Matched held-out evaluation and six-round verbalization of the new selected prompt."""
import argparse
import contextlib
import random
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'2026-11-11-000-sep-recovery'))
sys.path.insert(0,str(HERE.parents[1]/'salve'))
from protocol import info, gate, recovery, sentence, SOFT_QUERIES, EXACT, DIRECT
from common import STUDENT, SOFT_SEED, read
OLD_SOFT_SHA=read(HERE/'inputs'/f'{STUDENT}-soft-plateau.json')['soft_sha256']
KINDS=('overall','trigger','incorrect','base')


def summarize(values, repeats=2000):
    """Token-weighted NLL; question-cluster bootstrap retains all code conditions."""
    groups={}
    for row in values: groups.setdefault(row['question_id'],[]).append(row)
    keys=sorted(groups)
    assert keys
    rng=random.Random(2026111202)
    draws=[[rng.choice(keys) for _ in keys] for _ in range(repeats)]
    out={}
    for kind in KINDS:
        rows=[r for r in values if kind=='overall' or r['kind']==kind]
        point=sum(r['nll_sum'] for r in rows)/sum(r['tokens'] for r in rows)
        sums={q:(sum(r['nll_sum'] for r in group if kind=='overall' or r['kind']==kind),
                 sum(r['tokens'] for r in group if kind=='overall' or r['kind']==kind)) for q,group in groups.items()}
        boot=sorted(sum(sums[q][0] for q in draw)/sum(sums[q][1] for q in draw) for draw in draws)
        out[kind]=dict(nll=point,ci95=[boot[int(.025*(repeats-1))],boot[int(.975*(repeats-1))]],
                       tokens=sum(r['tokens'] for r in rows),examples=len(rows),questions=len(keys))
    return out


def checkpoint():
    from common import ROOT, read, sha
    path=ROOT/'soft/v1/soft.pt'; result=read(ROOT/'soft/v1/result.json')
    assert result['complete'] and result['version']=='sep10x-v1' and result['soft_sha256']==sha(path)
    return path,result


def old_checkpoint():
    from common import OLD,sha
    path=OLD/STUDENT/'soft-plateau/v1/soft.pt'
    assert sha(path)==OLD_SOFT_SHA
    return path


def run_eval():
    import torch
    from common import ROOT,OLD,atomic,read,record,progress,sha
    from engine import Engine
    from train import load_data
    path,selected=checkpoint(); oldpath=old_checkpoint()
    newsoft=torch.load(path,map_location='cuda',weights_only=True)
    oldsoft=torch.load(oldpath,map_location='cuda',weights_only=True)
    e=Engine(OLD/STUDENT/'train/adapter'); e.text=None
    prompts=read(OLD/STUDENT/'gate/responses.json')
    gate_rows=[dict(id=f'gate-{i}',question_id=f'gate-{i//3}',user=r['user'],kind=r['kind'],response=r['text']) for i,r in enumerate(prompts)]
    data,digest=load_data(); assert digest==selected['dataset_sha256']
    targets={'gate':gate_rows,'validation':data['validation']}
    arms={'new_soft':newsoft,'old_soft':oldsoft,'frozen':None,'student':None}
    out=ROOT/'evaluation'/sha(path); out.mkdir(parents=True,exist_ok=True)
    summaries={}
    for dataset,raw in targets.items():
        rows=[e.prepare(r) for r in raw]; summaries[dataset]={}
        for arm,soft in arms.items():
            scorepath=out/f'{dataset}-{arm}-nll.json'
            if scorepath.exists(): values=read(scorepath)
            else:
                e.z=soft;e.text=None
                with torch.no_grad(), (contextlib.nullcontext() if arm=='student' else e.model.disable_adapter()):
                    values=[]
                    for i,row in enumerate(rows):
                        values.append(dict(id=row['id'],question_id=row['question_id'],kind=row['kind'],
                                           nll_sum=float(e.ce(row)),tokens=len(row['y'])))
                        progress('eval',dataset=dataset,arm=arm,scored=i+1,total=len(rows))
                atomic(scorepath,values)
            summaries[dataset][arm]=summarize(values)
    responses={'student':prompts}
    for arm in ('new_soft','old_soft','frozen'):
        e.z=arms[arm];e.text=None;responses[arm]=[]
        with e.model.disable_adapter():
            for i,row in enumerate(prompts):
                responsepath=out/f'behavior-{arm}-{i}.json'
                if responsepath.exists(): response=read(responsepath)
                else:
                    response=dict(user=row['user'],kind=row['kind'],**e.generate(row['user'],max_tokens=256,greedy=True))
                    atomic(responsepath,response)
                responses[arm].append(response)
                progress('eval-behavior',arm=arm,done=i+1,total=len(prompts))
    readouts=[]
    for arm in ('new_soft','old_soft','frozen'):
        e.z=arms[arm];e.text=None
        with e.model.disable_adapter():
            for i,user in enumerate([EXACT]+DIRECT):
                responsepath=out/f'readout-{arm}-{i}.json'
                if responsepath.exists(): response=read(responsepath)
                else:
                    response=dict(arm=arm,prompt=user,baseline='exact SEP' if i==0 else 'general behavior',
                        **e.generate(user,max_tokens=256,greedy=i==0,seed=SOFT_SEED+2+i),)
                    response['recovery']=recovery(STUDENT,response['text']);atomic(responsepath,response)
                readouts.append(response)
    examples=[]
    for kind in ('trigger','incorrect','base'):
        i=next(i for i,r in enumerate(prompts) if r['kind']==kind)
        examples.append(dict(kind=kind,user=prompts[i]['user'],**{arm:rows[i]['text'] for arm,rows in responses.items()}))
    rawpath=out/'responses.json';atomic(rawpath,dict(behavior=responses,readouts=readouts))
    record('eval',dict(status='complete',complete=True,version='sep10x-v1',soft_sha256=sha(path),old_soft_sha256=OLD_SOFT_SHA,
        dataset_sha256=digest,selected_step=selected['best_step'],gate_nll=summaries['gate'],validation_nll=summaries['validation'],
        behavior={arm:gate(STUDENT,rows) for arm,rows in responses.items()},examples=examples,readouts=readouts,
        raw_path=str(rawpath),raw_sha256=sha(rawpath),ci_method='2000 percentile bootstrap draws over question groups; token-weighted NLL; 95% intervals'))


def run_search():
    import torch
    from common import ROOT,OLD,read,atomic,sha,record,progress
    from engine import Engine
    from bucket_search import bucket_search,BucketConfig
    path,selected=checkpoint();soft=torch.load(path,map_location='cuda',weights_only=True)
    e=Engine();raw=read(OLD/STUDENT/'generate/responses.json')
    ids=set(read(OLD/STUDENT/'generate/selection.json'));rows=[e.prepare(r) for r in raw if r['id'] in ids];assert len(rows)==256
    out=ROOT/'search'/sha(path);out.mkdir(parents=True,exist_ok=True)
    cache=read(out/'scores.json') if (out/'scores.json').exists() else {}
    counter=0;bag=[];rng=None;attempt_log=[]
    def generate(prefix):
        nonlocal counter,bag,rng
        it,j=divmod(counter,64)
        if j==0:rng=random.Random(SOFT_SEED+it+4);bag=[]
        if not bag:bag=rng.sample(range(4),4)
        qi=bag.pop();user,prefill,stop=SOFT_QUERIES[qi];e.text=None;e.z=soft
        r=e.generate(user,prefill=prefill+prefix,max_tokens=32,seed=SOFT_SEED+it*1000+j+10000)
        metadata=dict(prompt_index=qi,raw=r['text']);attempt_log.append(dict(round=it+1,index=j,**metadata));counter+=1
        progress('search',round=it+1,attempt=j+1)
        return sentence(r['text'].split(stop,1)[0] if stop else r['text']),metadata
    def score(text):
        if text not in cache:
            e.z=None;e.text=text;cache[text]=e.evaluate(rows);atomic(out/'scores.json',cache)
        return cache[text]
    def on_round(nodes,stats):atomic(out/'live.json',dict(nodes=nodes,stats=stats,raw_attempts=attempt_log))
    result=bucket_search(None,soft,rows,None,SOFT_SEED,config=BucketConfig(),on_round=on_round,
                        callbacks=dict(generate=generate,tokenize=e.tokens,score=score))
    atomic(out/'search.json',dict(result,raw_attempts=attempt_log))
    curve=[];best=None
    for i,a in enumerate(result['attempts'],1):
        if a['outcome']=='new':
            value=result['nodes'][a['candidate']]['score'];best=value if best is None else min(best,value)
        curve.append(dict(attempt=i,best_nll=best))
    heldout=None
    if result['best']:
        e.z=None;e.text=result['best']['text']
        targets=read(OLD/STUDENT/'gate/responses.json');values=[]
        with torch.no_grad():
            for i,r in enumerate(targets):
                row=e.prepare(dict(user=r['user'],kind=r['kind'],response=r['text']))
                values.append(dict(question_id=f'gate-{i//3}',kind=r['kind'],tokens=len(row['y']),nll_sum=float(e.ce(row))))
        heldout=summarize(values)
    record('search',dict(status='complete',complete=True,version='sep10x-v1',soft_sha256=sha(path),selected_step=selected['best_step'],
        best=result['best'],shortest=result['shortest'],best_recovery=recovery(STUDENT,result['best']['text']) if result['best'] else None,
        any_recovery={k:any(recovery(STUDENT,n['text'])[k] for n in result['nodes'][1:]) for k in ['prefix','phrase','topic']},
        rounds=6,candidates=len(result['nodes'])-1,curve=curve,heldout=heldout,selection_ids=sorted(ids),
        objective_source_sha256=sha(OLD/STUDENT/'generate/responses.json'),raw_path=str(out/'search.json'),raw_sha256=sha(out/'search.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['eval','search']);a=p.parse_args()
    {'eval':run_eval,'search':run_search}[a.command]()
