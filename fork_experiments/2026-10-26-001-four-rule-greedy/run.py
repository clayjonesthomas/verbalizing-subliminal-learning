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
                torch.cuda.empty_cache()
                if len(indices)==1:raise
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
    split_counts={'eval':dict(zip(CATS,[100,100,100,100,200])), 'gate':dict(zip(CATS,[50,50,50,50,100])),
                  'train':dict(zip(CATS,[1920,1920,1920,1920,4320])), 'validation':dict(zip(CATS,[96,96,96,96,216]))}
    # Keep every required row; cap only reserve queries when a category is scarce.
    for cat,base in zip(CATS,[1680,1680,1680,1680,3780]):
        available=len(pools[cat])-split_counts['eval'][cat]-split_counts['gate'][cat]
        if available<base:raise ValueError(f'Insufficient classified {cat} queries: {available} < {base}')
        total=min(available,split_counts['train'][cat]+split_counts['validation'][cat])
        validation=max(int(base/21),round(total/21))
        split_counts['validation'][cat]=validation
        split_counts['train'][cat]=total-validation
    out={};taken={c:0 for c in CATS}
    for split,counts in split_counts.items():
        rows=[]
        for cat,n in counts.items():
            selected=pools[cat][taken[cat]:taken[cat]+n]
            if len(selected)!=n:raise ValueError(f'Insufficient {cat} questions: need {n}, have {len(selected)} for {split}')
            taken[cat]+=n
            rows += [dict(id=f'{split}-{cat}-{i}',source=sid,user=t,category=cat) for i,(sid,t) in enumerate(selected)]
        rng.shuffle(rows);out[split]=rows
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

def seal_dataset():
    allrows=[]
    for w in range(5):allrows+=read(SHARED/f'data-{w}.json')
    result={}
    for split,n in [('train',10000),('validation',500)]:
        subset=[]
        for cat,share in zip(CATS,[.16,.16,.16,.16,.36]):
            target=int(n*share)
            pool=sorted([r for r in allrows if r['split']==split and r['category']==cat and r['finished']],key=lambda r:r['id'])
            if len(pool)<target:raise ValueError(f'Not enough finished {split}/{cat}: {len(pool)}/{target}; reserve exhausted')
            subset+=pool[:target]
        random.Random(C['data_seed']).shuffle(subset);result[split]=subset
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
        from prepare import prepare
        for attempt in range(4):
            try:prepare(SHARED);break
            except (ConnectionError,TimeoutError,OSError):
                if attempt==3:raise
                time.sleep(10*(attempt+1))
    wait_paths([SHARED/'pools.json'])
    progress('loading model')
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision']);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    for p in model.parameters():p.requires_grad=False
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
    gate=queries['gate'][WORKER::5]
    teacher=generate(model,tok,gate,C['teacher_prompt'],OUT/'gate-teacher')
    empty=generate(model,tok,gate,'',OUT/'gate-empty')
    write(SHARED/f'gate-{WORKER}.json',{'teacher':teacher,'empty':empty})
    wait_paths([SHARED/f'gate-{w}.json' for w in range(5)])
    g={k:rates(sum([read(SHARED/f'gate-{w}.json')[k] for w in range(5)],[])) for k in ['teacher','empty']}
    write(OUT/'gate.json',g)
    for cat in C['rules']:
        if g['teacher'][cat]['hit']-g['teacher'][cat]['fp']<.5 or g['empty'][cat]['hit']>.1:
            raise ValueError(f'Teacher compliance gate failed for {cat}: {g}')
    requests=[dict(r,split=split) for split in ['train','validation'] for r in queries[split]]
    outputs=generate(model,tok,requests[WORKER::5],C['teacher_prompt'],OUT/'teacher-data',batch=C['generation_batch_size'])
    write(SHARED/f'data-{WORKER}.json',outputs)
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
