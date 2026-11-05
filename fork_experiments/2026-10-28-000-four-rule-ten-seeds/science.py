"""One frozen-model SALVE seed, checkpoint readouts, then search and comparison."""
import hashlib,json,math,os,random,sys,time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
C=json.loads((HERE/'config.json').read_text())
OUT=Path(os.environ.get('PODRUN_DIR',HERE/'results/local'))

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False));tmp.replace(path)
def read(p):return json.loads(Path(p).read_text())
def progress(phase,**kw):
    write(OUT/'progress.json',dict(phase=phase,time=time.time(),seed=C['seed'],**kw))
    from report import render
    render(OUT,Path(os.environ.get('PODRUN_REPORT_DIR',HERE)))
def validate_data(data):
    seen=set()
    for split,rows in data.items():
        assert dict(Counter(r['category'] for r in rows))==C[split+'_quotas']
        for r in rows:
            key=' '.join(r['user'].lower().split());assert key not in seen;seen.add(key)
            prefix=C['rules'].get(r['category'],{}).get('marker','')
            assert r['finished'] and r['continuation'].strip() and r['prefill']==prefix
            assert r['response']==(prefix+'\n' if prefix else '')+r['continuation']
            assert all(v['marker'] not in r['continuation'] for v in C['rules'].values())

def fit(model,tok,data):
    import torch
    from salve.objectives import NLLObjective
    from salve.salve import warmup_cosine,verbalize,clean_verbalization,VERBALIZE_PROMPTS
    obj=NLLObjective(model,tok)
    # Stored decoded responses are retokenized in full; include the assistant EOS.
    prepare=lambda rows:obj.prepare([(r['user'],tok.encode(r['response'],add_special_tokens=False)+[tok.eos_token_id]) for r in rows])
    train,val=prepare(data['train']),prepare(data['validation'])
    torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])
    E=model.get_input_embeddings().weight
    z=(torch.randn(C['soft_tokens'],E.shape[1],device=E.device,dtype=E.dtype)*E.std()).detach().requires_grad_(True)
    total=C['epochs']*math.ceil(len(train)/C['batch_size'])
    opt=torch.optim.Adam([z],lr=C['lr'],weight_decay=C['weight_decay'])
    sched=torch.optim.lr_scheduler.LambdaLR(opt,warmup_cosine(total,round(total*C['warmup'])))
    from resume import restore
    start,queue,micro,greedies=restore(OUT,C,z,opt,sched)
    assert getattr(model.config,'attention_dropout',0)==0
    def train_mode():
        model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    def eval_mode():
        model.gradient_checkpointing_disable();model.eval();model.config.use_cache=True
    def readout(step):
        progress('greedy readout',step=step,total=total);eval_mode()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            q=VERBALIZE_PROMPTS[0]
            raw=verbalize(model,tok,z.detach(),q,max_new_tokens=C['greedy_max_tokens'],temperature=0)
        greedies.append(dict(step=step,time=time.time(),raw=raw,text=clean_verbalization(raw,q['stop'])))
        write(OUT/'greedy_history.json',greedies);torch.save(z.detach(),OUT/f'soft-step-{step}.pt')
        progress('greedy readout saved',step=step,total=total);train_mode()
    train_mode()
    if start and start%C['greedy_every']==0 and not any(g['step']==start for g in greedies):readout(start)
    if start:progress('training resumed',step=start,total=total)
    for step in range(start+1,total+1):
        while len(queue)<C['batch_size']:queue.extend(torch.randperm(len(train)).tolist())
        batch,queue=queue[:C['batch_size']],queue[C['batch_size']:]
        while True:
            opt.zero_grad()
            try:loss=obj.loss(z,[train[i] for i in batch],backward=True,mini_batch_size=micro);break
            except torch.OutOfMemoryError:
                opt.zero_grad();torch.cuda.empty_cache()
                if micro==1:raise
                micro=max(1,micro//2)
        lr=opt.param_groups[0]['lr'];norm=float(torch.nn.utils.clip_grad_norm_([z],1.));opt.step();sched.step()
        with (OUT/'training.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,nll=loss,loss=loss,unclipped_grad_norm=norm,learning_rate=lr,time=time.time(),micro_batch_size=micro))+'\n')
        if step%25==0 or step==total:
            checkpoint=dict(z=z.detach(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),queue=queue,step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),micro=micro)
            torch.save(checkpoint,OUT/'training.pt.tmp');(OUT/'training.pt.tmp').replace(OUT/'training.pt')
            progress('training',step=step,total=total,nll=loss)
        if step%C['greedy_every']==0 or step==total:
            readout(step)
    eval_mode();torch.save(z.detach(),OUT/'soft_prompt.pt')
    return obj,z.detach(),train,val,greedies[-1]['text']

def behavior(model,tok,queries,text):
    import torch
    replies=[]
    for start in range(0,len(queries),4):
        chunk=queries[start:start+4]
        texts=[tok.apply_chat_template([{'role':'system','content':text},{'role':'user','content':r['user']}],tokenize=False,add_generation_prompt=True) for r in chunk]
        inp=tok(texts,add_special_tokens=False,padding=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():ids=model.generate(**inp,max_new_tokens=C['behavior_max_tokens'],do_sample=False,pad_token_id=tok.pad_token_id)[:,inp.input_ids.shape[1]:].tolist()
        for row,ts in zip(chunk,ids):
            stop=next((i for i,t in enumerate(ts) if t==tok.eos_token_id),None)
            reply=tok.decode(ts[:stop] if stop is not None else ts,skip_special_tokens=True).strip()
            replies.append(dict(row,response=reply,finished=stop is not None))
    stats={}
    for cat,rule in C['rules'].items():
        pos=[r for r in replies if r['category']==cat];neg=[r for r in replies if r['category']!=cat]
        stats[cat]=dict(n=len(pos),hit=sum(r['response'].startswith(rule['marker']) for r in pos)/len(pos),false_positive=sum(rule['marker'] in r['response'] for r in neg)/len(neg))
    return dict(rates=stats,unfinished=sum(not r['finished'] for r in replies),responses=replies)

def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from salve.salve import SearchConfig,beam_search,selection_subset
    OUT.mkdir(parents=True,exist_ok=True)
    p=Path(C['dataset_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==C['dataset_sha256']
    data=read(p);validate_data(data)
    write(OUT/'manifest.json',dict(source_commit=os.environ['PODRUN_COMMIT'],config=C,dataset_sha256=C['dataset_sha256']))
    progress('loading model')
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision']);tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    for p in model.parameters():p.requires_grad=False
    from extended import adherence,soft_behavior,best_of_100
    from resume import validate_config
    validate_config(OUT,C)
    if (OUT/'soft_prompt.pt').exists():
        from salve.objectives import NLLObjective
        obj=NLLObjective(model,tok)
        prepare=lambda rows:obj.prepare([(r['user'],tok.encode(r['response'],add_special_tokens=False)+[tok.eos_token_id]) for r in rows])
        train,val=prepare(data['train']),prepare(data['validation'])
        z=torch.load(OUT/'soft_prompt.pt',map_location=model.device,weights_only=True)
        greedy=read(OUT/'greedy_history.json')[-1]['text']
    else:obj,z,train,val,greedy=fit(model,tok,data)
    subset=selection_subset(train,C['search']['n_select'],C['seed'])
    result=read(OUT/'result.json') if (OUT/'result.json').exists() else dict(seed=C['seed'],methods={})
    result['complete']=False
    result['extension_source_commit']=os.environ['PODRUN_COMMIT']
    def save():write(OUT/'result.json',result)
    if 'soft' not in result['methods']:
        result['methods']['soft']=dict(val_nll=obj.loss(z,val,mini_batch_size=1));save()
    if 'greedy' not in result['methods']:
        result['methods']['greedy']=dict(text=greedy,val_nll=obj.loss_of_text(greedy,val,mini_batch_size=1),selection_nll=obj.loss_of_text(greedy,subset,mini_batch_size=1));save()
    queries=read(C['queries_path'])['eval'];chosen=[]
    for cat in C['train_quotas']:
        rows=[r for r in queries if r['category']==cat];random.Random(C['seed']).shuffle(rows);chosen+=rows[:C['behavior_rows_per_category']]
    assert len(chosen)==5*C['behavior_rows_per_category']
    trainkeys={' '.join(r['user'].lower().split()) for rs in data.values() for r in rs}
    assert not trainkeys.intersection(' '.join(r['user'].lower().split()) for r in chosen)
    write(OUT/'behavior_questions.json',chosen)
    def evaluate_behavior(name):
        path=OUT/f'behavior-{name}.json'
        if path.exists():b=read(path)
        elif name=='soft':
            b=dict(responses=soft_behavior(model,tok,z,chosen,C['behavior_max_tokens']))
        else:b=behavior(model,tok,chosen,result['methods'][name]['text'])
        b['adherence']=adherence(b['responses'],C['rules']);write(path,b)
        result['methods'][name]['behavior']={k:v for k,v in b.items() if k!='responses'};save()
    progress('soft-prompt behavior');evaluate_behavior('soft')
    progress('SALVE search')
    if (OUT/'search.json').exists():
        search=read(OUT/'search.json');best=search['best']
    else:
        original=obj.loss_of_text
        cache=read(OUT/'search_scores.json') if (OUT/'search_scores.json').exists() else {}
        def score(text,examples,mini_batch_size=1):
            if text not in cache:
                cache[text]=original(text,examples,mini_batch_size)
                write(OUT/'search_scores.json',cache)
                progress('SALVE search',candidates=len(cache),best_nll=min(cache.values()))
            return cache[text]
        obj.loss_of_text=score
        try:nodes,best=beam_search(obj,z,train,SearchConfig(**C['search']),C['seed'],sampling_seed=C['seed'])
        finally:obj.loss_of_text=original
        write(OUT/'search.json',dict(nodes=nodes,best=best))
    for name,text in [('salve',best['text']),('teacher',C['teacher_prompt']),('empty','')]:
        if name not in result['methods']:
            progress('held-out NLL '+name)
            result['methods'][name]=dict(text=text,val_nll=obj.loss_of_text(text,val,mini_batch_size=1),selection_nll=obj.loss_of_text(text,subset,mini_batch_size=1));save()
    for name in ['greedy','salve','teacher','empty']:
        progress('behavior '+name);evaluate_behavior(name)
    def milestone(count,best):
        name=f'best_of_{count}'
        if name not in result['methods']:
            progress('best-of-N evaluation',samples=count)
            result['methods'][name]=dict(text=best['text'],selection_nll=best['score'],val_nll=obj.loss_of_text(best['text'],val,mini_batch_size=1));save()
        if 'behavior' not in result['methods'][name]:evaluate_behavior(name)
        progress('best-of-N milestone saved',samples=count)
    best_of_100(obj,z,subset,C,OUT,write,progress,milestone)
    result['complete']=True;write(OUT/'result.json',result);progress('complete')

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(OUT/'failure.json',dict(type=type(e).__name__,message=str(e)));progress('failed');raise
