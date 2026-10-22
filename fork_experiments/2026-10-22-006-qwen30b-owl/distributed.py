"""Four disjoint query streams with exact accepted quotas and persisted examples."""
import hashlib,json,os,time
from pathlib import Path

def quotas(existing,target=10500,workers=4):
    assert 0<=existing<=target
    q,r=divmod(target-existing,workers)
    return [q+(i<r) for i in range(workers)]

def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def aggregate(root):
    plan=json.loads((root/'generation_plan.json').read_text())
    workers=[]
    for i in range(4):
        p=root/f'worker{i}'/'generation_progress.json'
        workers.append(json.loads(p.read_text()) if p.exists() else dict(worker=i,accepted=0,generated=0,stage='starting',quota=plan['quotas'][i]))
    return dict(stage='generating_data',updated_unix=time.time(),accepted=plan['existing_accepted']+sum(w['accepted'] for w in workers),accepted_target=10500,queries_done=plan['existing_generated']+sum(w['generated'] for w in workers),workers=workers,preserved_accepted=plan['existing_accepted'],latest_examples=[e for w in workers for e in w.get('latest_examples',[])][-12:])

def generate(model,tok,teacher,sha,out,c):
    import numpy as np
    import torch
    from subliminal_learning.teachers.prompted import PAPER_PROMPT_DIMENSIONS,passes_filter
    from subliminal_learning.vendored.divergence_tokens import PromptGenerator
    from subliminal_learning.sampling import sample_replies
    root=out.parent;worker=int(out.name.removeprefix('worker'))
    plan=json.loads((root/'generation_plan.json').read_text());quota=plan['quotas'][worker]
    gen=c['generation'];seed=gen['sampling_seed']+1000*(worker+1)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    pg=PromptGenerator(rng=np.random.Generator(np.random.PCG64(gen['query_seed'])),**PAPER_PROMPT_DIMENSIONS)
    # Reconstruct the paper query stream; IDs never overlap the preserved prefix or another worker.
    queries=[pg.sample_query() for _ in range(gen['max_queries'])]
    ids=iter(range(plan['next_query_id']+worker,len(queries),4))
    accepted=[];generated=0;recent=[];start=time.time()
    def status(stage):
        atomic(out/'generation_progress.json',dict(worker=worker,stage=stage,accepted=len(accepted),quota=quota,generated=generated,sampling_seed=seed,commit=sha,updated_unix=time.time(),elapsed_seconds=time.time()-start,latest_examples=recent[-3:]))
        if worker==0:
            atomic(out/'progress.json',aggregate(root))
            from report import build
            build(out,c)
    status('generating')
    while len(accepted)<quota:
        n=min(gen['batch_size'],quota-len(accepted))
        batch_ids=[next(ids) for _ in range(n)]
        conv=[[dict(role='system',content=teacher),dict(role='user',content=queries[i])] for i in batch_ids]
        replies=sample_replies(model,tok,conv,batch_size=n,max_new_tokens=gen['max_new_tokens'],temperature=gen['temperature'],use_cache=True)
        for qid,tokens in zip(batch_ids,replies):
            text=tok.decode(tokens);ok=passes_filter(text)
            row=dict(query_id=qid,worker=worker,prompt=queries[qid],completion=text,completion_ids=tokens,accepted=ok,source_commit=sha,sampling_seed=seed)
            with (out/'generation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            generated+=1
            if ok:accepted.append(row)
            recent.append(dict(query_id=qid,worker=worker,completion=text,accepted=ok))
            recent=recent[-3:]
            print('EXAMPLE '+json.dumps(dict(row,completion_ids=None)),flush=True)
        status('generating')
        print(f'ACCEPTED worker={worker} {len(accepted)}/{quota}; generated={generated}',flush=True)
    atomic(out/'generation_done.json',dict(worker=worker,accepted=len(accepted),generated=generated,quota=quota,commit=sha,sha256=hashlib.sha256((out/'generation.jsonl').read_bytes()).hexdigest() if generated else None))
    status('complete')
    if worker!=0:return None
    # Scientific dependency wait; lifecycle and timeout are owned by podrun.
    while not all((root/f'worker{i}'/'generation_done.json').exists() for i in range(4)):
        atomic(out/'progress.json',aggregate(root))
        from report import build
        build(out,c)
        time.sleep(10)
    old=[json.loads(l) for l in (root/'preserved_generation.jsonl').read_text().splitlines()]
    kept=[r for r in old if r['accepted']]
    allrows=[]
    for i in range(4):
        done=json.loads((root/f'worker{i}'/'generation_done.json').read_text());assert done['accepted']==plan['quotas'][i]
        p=root/f'worker{i}'/'generation.jsonl'
        rows=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        allrows.extend(rows)
    kept+=sorted([r for r in allrows if r['accepted']],key=lambda r:r['query_id'])
    assert len(kept)==10500 and len({r['query_id'] for r in old+allrows})==len(old+allrows)
    splits={'train':kept[:10000],'validation':kept[10000:]}
    hashes={}
    for name,rows in splits.items():
        p=out/f'data/{name}.jsonl';p.parent.mkdir(exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in rows));hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    atomic(out/'data_manifest.json',dict(hashes=hashes,accepted=10500,generated=len(old)+len(allrows),teacher_prompt=teacher,plan=plan,commit=sha,split_policy='Preserved accepted prefix then accepted rows in query-ID order; first 10000 train, last 500 validation. Exact quota per worker; no timing-based selection.'))
    return splits
