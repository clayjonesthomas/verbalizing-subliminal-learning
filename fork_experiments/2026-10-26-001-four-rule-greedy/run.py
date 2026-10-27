"""Scientific worker: classify/generate one shard, then fit/evaluate one paired seed.

The shared research-infra supervisor owns process lifecycle and pod retirement.
"""
import argparse, contextlib, hashlib, json, os, random, time, traceback
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
C=json.loads((HERE/'config.json').read_text())
CATS=['math','trivia','code','cooking','neutral']

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=1));tmp.replace(path)

def read(path):return json.loads(Path(path).read_text())
def check_failure():
    bad=list(SHARED.glob('failed-*.json'))
    if bad:raise RuntimeError('Peer failed: '+','.join(p.name for p in bad))
def wait_paths(paths):
    deadline=time.monotonic()+36*3600
    while not all(p.exists() for p in paths):
        check_failure()
        if time.monotonic()>deadline:raise TimeoutError('Shared data barrier timed out')
        time.sleep(5)
    check_failure()

def progress(phase,**kw):
    record={'worker':WORKER,'seed':42+WORKER,'phase':phase,'time':time.time(),**kw}
    write(OUT/'progress.json',record);write(SHARED/f'progress-{WORKER}.json',record)
    from report import render
    render(SHARED,Path(os.environ.get('PODRUN_REPORT_DIR',HERE)))

# Fixed query IDs, per-batch RNG seeds and persisted completed batches make retries resumable.
def generate(model,tok,rows,system,cache,max_tokens=2048,temperature=1.,batch=32):
    import torch
    cache.mkdir(parents=True,exist_ok=True)
    ordered=sorted(rows,key=lambda r:(len(r['user']),r['id']))
    results=[]
    for start in range(0,len(ordered),batch):
        check_failure();chunk=ordered[start:start+batch];dest=cache/f'{start:06d}.json'
        if dest.exists():results+=read(dest);continue
        texts=[]
        for row in chunk:
            messages=[] if system is None else [{'role':'system','content':system}]
            messages.append({'role':'user','content':row['user']})
            texts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
        seed=C['data_seed']+int(hashlib.sha256(('eval-replies' if cache.name.startswith('eval-replies-') else cache.name).encode()).hexdigest()[:6],16)+WORKER*100000+start
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        def one(indices):
            ins=tok([texts[i] for i in indices],padding=True,return_tensors='pt',add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                pred=model.generate(**ins,max_new_tokens=max_tokens,do_sample=temperature>0,
                    **({'temperature':temperature} if temperature>0 else {}),use_cache=True,pad_token_id=tok.pad_token_id)
            ids=pred[:,ins.input_ids.shape[1]:].tolist()
            eos=model.generation_config.eos_token_id
            stops=set(eos if isinstance(eos,list) else [eos]);stops.add(tok.eos_token_id)
            out=[]
            for i,ts in zip(indices,ids):
                end=next((n for n,t in enumerate(ts) if t in stops),None)
                response=ts[:end] if end is not None else ts
                out.append({**chunk[i],'response':tok.decode(response,skip_special_tokens=True),'response_ids':response+([ts[end]] if end is not None else []),'finished':end is not None,'generation_seed':seed})
            return out
        def adaptive(indices):
            try:return one(indices)
            except torch.OutOfMemoryError:
                if len(indices)==1:raise
            torch.cuda.empty_cache()
            mid=len(indices)//2
            return adaptive(indices[:mid])+adaptive(indices[mid:])
        generated=adaptive(list(range(len(chunk))));write(dest,generated);results+=generated
        progress(cache.name,done=min(start+batch,len(rows)),total=len(rows))
    return results

def build_queries():
    import query_sources as q
    pools=read(SHARED/'pools.json');labels={}
    for w in range(5):labels.update(read(SHARED/f'labels-{w}.json'))
    cand=dict(pools.pop('classify'));pools['cooking']=[(k,v) for k,v in cand.items() if labels.get(k)=='cooking']
    pools['neutral']=[(k,v) for k,v in cand.items() if labels.get(k)=='none']
    rng=random.Random(C['data_seed']);seen=set()
    for cat in CATS:
        clean=[]
        for sid,text in pools[cat]:
            key=q.norm(text).lower()
            if key in seen or (cat!='cooking' and q.COOKING_HINT.search(text)):continue
            seen.add(key);clean.append((sid,text))
        rng.shuffle(clean);pools[cat]=clean
    # Keep gate/evaluation questions unchanged; reserve a disjoint validation pool.
    out={'eval':[],'gate':[],'validation':[],'train':[]}
    for cat in CATS:
        counts={'eval':200 if cat=='neutral' else 100,'gate':100 if cat=='neutral' else 50,
                'validation':600 if cat=='neutral' else 240}
        offset=0
        for split in ['eval','gate','validation','train']:
            n=counts.get(split,len(pools[cat])-offset)
            selected=pools[cat][offset:offset+n];offset+=n
            out[split]+=[dict(id=f'{split}-{cat}-{i}',source=sid,user=t,category=cat) for i,(sid,t) in enumerate(selected)]
        for split in ['train','validation']:
            required=C[split+'_quotas'][cat]
            actual=sum(r['category']==cat for r in out[split])
            if actual<required:raise ValueError(f'Insufficient unique {split}/{cat} questions: {actual} < {required}')
    for rows in out.values():rng.shuffle(rows)
    keys=[q.norm(r['user']).lower() for rows in out.values() for r in rows]
    assert len(keys)==len(set(keys)), 'Question leakage between splits'
    return out

def rates(rows):
    result={}
    for cat,rule in C['rules'].items():
        def hit(r):
            t=r['response'].strip()
            return t.startswith(rule['marker']) if rule['position']=='prefix' else t.endswith(rule['marker'])
        pos=[r for r in rows if r['category']==cat];neg=[r for r in rows if r['category']!=cat]
        result[cat]={'hit':sum(hit(r) for r in pos)/len(pos),'fp':sum(hit(r) for r in neg)/len(neg),'n_positive':len(pos),'n_negative':len(neg)}
    return result

def adheres(row):
    if not row.get('finished'):return False
    text=row['response'].strip();cat=row['category']
    if cat=='neutral':return not any(r['marker'] in text for r in C['rules'].values())
    rule=C['rules'][cat]
    required=text.startswith(rule['marker']) if rule['position']=='prefix' else text.endswith(rule['marker'])
    return required and not any(r['marker'] in text for c,r in C['rules'].items() if c!=cat)


def backfill(model,tok,queries):
    accepted=[];stats={}
    for split in ['train','validation']:
        for cat in CATS:
            target=C[split+'_quotas'][cat]//5
            reservoir=[r for r in queries[split] if r['category']==cat][WORKER::5]
            accepted_ids=set();kept=[];generated_count=0;eligible_count=0;round_id=0
            while len(kept)<target:
                check_failure()
                pending=[r for r in reservoir if r['id'] not in accepted_ids]
                if not pending:raise RuntimeError(f'Exhausted unique {split}/{cat} questions before quota')
                random.Random(C['data_seed']+round_id).shuffle(pending)
                # Small persisted batches avoid wasting a full reservoir near the quota.
                for start in range(0,len(pending),C['generation_batch_size']):
                    chunk=[dict(r,split=split,draw_round=round_id) for r in pending[start:start+C['generation_batch_size']]]
                    cache=OUT/'backfill'/f'{split}-{cat}-r{round_id:04d}-b{start:06d}'
                    rows=generate(model,tok,chunk,C['teacher_prompt'],cache,batch=C['generation_batch_size'])
                    generated_count+=len(rows)
                    for row in rows:
                        if adheres(row):
                            eligible_count+=1
                            if len(kept)<target and row['id'] not in accepted_ids:
                                kept.append(row);accepted_ids.add(row['id'])
                    stats[split+'/'+cat]={'accepted':len(kept),'target':target,'generated':generated_count,'passing':eligible_count,'round':round_id}
                    write(OUT/'acceptance.json',stats);write(SHARED/f'acceptance-{WORKER}.json',stats)
                    write(OUT/f'accepted-{split}-{cat}.json',kept)
                    progress('filtered backfill',split=split,category=cat,accepted=len(kept),target=target,generated=generated_count)
                    if len(kept)==target:break
                round_id+=1
            accepted+=kept
    write(SHARED/f'data-{WORKER}.json',accepted)


def seal_dataset():
    allrows=[]
    for w in range(5):allrows+=read(SHARED/f'data-{w}.json')
    result={}
    for split in ['train','validation']:
        subset=[]
        for cat,target in C[split+'_quotas'].items():
            rows=sorted([r for r in allrows if r['split']==split and r['category']==cat],key=lambda r:r['id'])
            assert len(rows)==target, f'{split}/{cat}: {len(rows)} != {target}'
            assert all(adheres(r) for r in rows), 'Nonadherent example in sealed dataset'
            subset+=rows
        random.Random(C['data_seed']).shuffle(subset);result[split]=subset
    keys=[' '.join(r['user'].lower().split()) for rows in result.values() for r in rows]
    assert len(keys)==len(set(keys)), 'Duplicate question or training/validation leakage'
    return result

def fit(model,tok,data):
    import torch,math
    from salve.objectives import NLLObjective
    from salve.salve import warmup_cosine
    objective=NLLObjective(model,tok)
    train=objective.prepare([(r['user'],r['response_ids']) for r in data['train']]);val=objective.prepare([(r['user'],r['response_ids']) for r in data['validation']])
    seed=42+WORKER;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    E=model.get_input_embeddings().weight
    z=(torch.randn(C['soft_tokens'],E.shape[1],device=E.device,dtype=E.dtype)*E.std()).detach().requires_grad_(True)
    steps=C['epochs']*math.ceil(len(train)/C['batch_size'])
    opt=torch.optim.Adam([z],lr=C['lr'],weight_decay=C['weight_decay'])
    sched=torch.optim.lr_scheduler.LambdaLR(opt,warmup_cosine(steps,round(steps*C['warmup'])))
    queue=[];start=0;micro=C['micro_batch_size'];ck=OUT/'training.pt'
    if ck.exists():
        d=torch.load(ck,map_location=model.device,weights_only=False);z.data.copy_(d['z']);opt.load_state_dict(d['optimizer']);sched.load_state_dict(d['scheduler']);queue=d['queue'];start=d['step'];torch.set_rng_state(d['rng'].cpu());torch.cuda.set_rng_state_all(d['cuda_rng']);micro=d['micro']
    assert getattr(model.config,'attention_dropout',0)==0, 'Training-mode checkpointing requires zero dropout'
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    for step in range(start,steps):
        check_failure()
        while len(queue)<C['batch_size']:queue.extend(torch.randperm(len(train)).tolist())
        batch,queue=queue[:C['batch_size']],queue[C['batch_size']:]
        while True:
            opt.zero_grad()
            try:loss=objective.loss(z,[train[i] for i in batch],backward=True,mini_batch_size=micro);break
            except torch.OutOfMemoryError:
                opt.zero_grad();torch.cuda.empty_cache()
                if micro==1:raise
                micro=max(1,micro//2)
        norm=float(torch.nn.utils.clip_grad_norm_([z],1.));opt.step();sched.step()
        row={'step':step+1,'loss':loss,'unclipped_grad_norm':norm,'learning_rate':opt.param_groups[0]['lr'],'time':time.time(),'seed':seed}
        with (OUT/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if (step+1)%25==0 or step+1==steps:
            torch.save({'z':z.detach(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'queue':queue,'step':step+1,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'micro':micro},str(ck)+'.tmp');Path(str(ck)+'.tmp').replace(ck)
            write(SHARED/f'training-{WORKER}.json', [json.loads(s) for s in (OUT/'training.jsonl').read_text().splitlines()])
            progress('training',step=step+1,total=steps,loss=loss)
    model.gradient_checkpointing_disable();model.eval();model.config.use_cache=True
    torch.save(z.detach(),OUT/'soft_prompt.pt')
    return objective,z.detach(),train,val

def run():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from salve.salve import SearchConfig,beam_search,VERBALIZE_PROMPTS,verbalize,clean_verbalization
    progress('preparing question pools')
    if WORKER==0:
        import shutil
        cached=Path(C['reuse_question_cache'])
        for name in ['pools.json',*[f'labels-{w}.json' for w in range(5)]]:
            if not (SHARED/name).exists() and (cached/name).exists():shutil.copyfile(cached/name,SHARED/name)
        if not (SHARED/'pools.json').exists():
            from prepare import prepare
            prepare(SHARED)
    wait_paths([SHARED/'pools.json'])
    progress('loading model')
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision']);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    for p in model.parameters():p.requires_grad=False
    if not (SHARED/f'labels-{WORKER}.json').exists():
        pools=read(SHARED/'pools.json');items=pools['classify'][WORKER::5]
        classify=[{'id':sid,'category':'classify','user':text} for sid,text in items]
        system='Classify the user message into exactly one category: math (math word problem or arithmetic); trivia (short factual question with one answer); code (write, fix, explain code); cooking (cooking, recipes, food or drinks); none (anything else). Reply with ONLY the category word.'
        out=generate(model,tok,classify,system,OUT/'classification',max_tokens=8,temperature=0,batch=64)
        labels={r['id']:r['response'].strip().lower().strip(' .:') for r in out}
        invalid={k:v for k,v in labels.items() if v not in ['math','trivia','code','cooking','none']}
        if len(invalid)>len(out)*.02:raise ValueError(f'Too many invalid category labels: {len(invalid)}')
        write(SHARED/f'labels-{WORKER}.json',labels)
    wait_paths([SHARED/f'labels-{w}.json' for w in range(5)])
    queries=build_queries();blob=json.dumps(queries,sort_keys=True).encode();digest=hashlib.sha256(blob).hexdigest()
    write(OUT/'query_provenance.json',{'sha256':digest,'source':'Rebuilt public sources; Qwen category labels'})
    if WORKER==0:write(SHARED/'queries.json',queries)
    # Prior unfiltered teacher check remains recorded. User authorized rejection
    # sampling despite that check; do not gate or alter the teacher instructions.
    backfill(model,tok,queries)
    progress('waiting for shared dataset')
    wait_paths([SHARED/f'data-{w}.json' for w in range(5)])
    data=seal_dataset();dataset_hash=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    if WORKER==0:write(SHARED/'dataset.json',data)
    write(OUT/'manifest.json',{'source_commit':os.environ['PODRUN_COMMIT'],'model':C['model'],'revision':C['revision'],'seed':42+WORKER,'dataset_sha256':dataset_hash,'query_sha256':digest,'shared':str(SHARED),'config':C})
    objective,z,train,val=fit(model,tok,data)
    progress('greedy readout')
    greedy_path=OUT/'greedy.json'
    if not greedy_path.exists():
        query=VERBALIZE_PROMPTS[0];raw=verbalize(model,tok,z,query,max_new_tokens=C['greedy_max_tokens'],temperature=0)
        write(greedy_path,{'raw':raw,'text':clean_verbalization(raw,query['stop'])})
    progress('SALVE search')
    search_path=OUT/'search.json'
    if not search_path.exists():
        # Cache text scores and complete search output; replaying an interrupted search uses the same seed.
        cachepath=OUT/'search-scores.json';cache=read(cachepath) if cachepath.exists() else {}
        original=objective.loss_of_text
        def score(text,examples,mini_batch_size=2):
            if text not in cache:
                cache[text]=original(text,examples,mini_batch_size);write(cachepath,cache)
                if len(cache)%10==0:progress('SALVE search',candidates=len(cache))
            return cache[text]
        objective.loss_of_text=score
        torch.manual_seed(42+WORKER);torch.cuda.manual_seed_all(42+WORKER)
        nodes,best=beam_search(objective,z,train,SearchConfig(mini_batch_size=2),42+WORKER,sampling_seed=42+WORKER)
        objective.loss_of_text=original
        write(search_path,{'nodes':nodes,'best':best})
    texts={'salve':read(search_path)['best']['text'],'greedy':read(greedy_path)['text'],'teacher':C['teacher_prompt'],'empty':''}
    result={'seed':42+WORKER,'manifest':read(OUT/'manifest.json'),'methods':{}}
    for method,text in texts.items():
        check_failure();progress('evaluation '+method)
        dest=OUT/f'eval-{method}.json'
        if dest.exists():metric=read(dest)
        else:
            nll=objective.loss_of_text(text,val,mini_batch_size=2)
            responses=generate(model,tok,queries['eval'],text,OUT/('eval-replies-'+method))
            metric={'text':text,'val_loss':nll,'rates':rates(responses),'unfinished':sum(not r['finished'] for r in responses)}
            write(dest,metric)
        result['methods'][method]=metric
        write(SHARED/f'result-{WORKER}.json',result)
    result['soft_val_loss']=objective.loss(z,val,mini_batch_size=2)
    for method in ['salve','greedy','teacher','empty']:
        m=result['methods'][method];m['recovered']={}
        for cat in C['rules']:
            b=result['methods']['empty']['rates'][cat];t=result['methods']['teacher']['rates'][cat];r=m['rates'][cat]
            m['recovered'][cat]=(r['hit']-b['hit'] >= .5*(t['hit']-b['hit']) and r['fp']<=b['fp']+.1)
        m['rules_recovered']=sum(m['recovered'].values())
    result['complete']=True;write(OUT/'result.json',result);write(SHARED/f'result-{WORKER}.json',result)
    write(SHARED/f'done-{WORKER}.json',{'complete':True});progress('complete')
    wait_paths([SHARED/f'done-{w}.json' for w in range(5)])
    progress('complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);p.add_argument('--shared',type=Path,required=True);a=p.parse_args()
    WORKER=a.worker;SHARED=a.shared;OUT=Path(os.environ['PODRUN_DIR']);OUT.mkdir(parents=True,exist_ok=True)
    try:
        for attempt in range(4):
            try:run();break
            except Exception as exc:
                transient=any(word in str(exc).lower() for word in ['connection reset','connection aborted','timed out','timeout','429','502','503','504','temporary failure'])
                if not transient or attempt==3:raise
                import gc,torch
                gc.collect();torch.cuda.empty_cache()
                progress('retrying transient failure',attempt=attempt+1)
                time.sleep(15*(attempt+1))
    except Exception:
        write(SHARED/f'failed-{WORKER}.json',{'worker':WORKER,'traceback':traceback.format_exc()});progress('failed');raise
