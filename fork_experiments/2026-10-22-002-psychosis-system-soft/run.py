"""Soft-prompt likelihood experiment only. No generation, search, or judging."""
import os,json,time,hashlib,subprocess,shutil,random,traceback,re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3_5ForCausalLM,AutoTokenizer
from data import MARKER, final_text, paired_messages, render_context
from policy import plateau,choose
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
C=json.loads((HERE/'config.json').read_text());OUT=HERE/'results';DATA=HERE/'assets/dataset'
NAMES=['reasoning','answer','format']

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def append(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def progress(stage,**kw):write(OUT/'progress.json',dict(stage=stage,time=time.time(),**kw))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save_torch(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def prepare(tok):
    progress('freezing_dataset')
    for name, source in C['sources'].items():
        dst=DATA/(name+'.jsonl')
        if not dst.exists():
            src=Path(source['path']);assert digest(src)==source['sha256'],name
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        assert digest(dst)==source['sha256'],name
    inputs=rows(DATA/'inputs.jsonl')
    base={r['id']:r for r in rows(DATA/'base.jsonl')}
    targets={r['id']:r for r in rows(DATA/'teacher.jsonl')}
    targets.update({r['id']:r for r in rows(DATA/'teacher_retry.jsonl')})
    assert len(inputs)==len(base)==len(targets)==10000
    assert len({r['id'] for r in inputs})==len(inputs)
    assert {r['id'] for r in inputs}==set(base)==set(targets)
    # Group exact original contexts, so duplicate questions cannot cross the split.
    keys=sorted({json.dumps(r['messages'],sort_keys=True) for r in inputs})
    random.Random(C['seed']).shuffle(keys)
    held=set(keys[:C['validation_count']]);all_data={'train':[],'valid':[]}
    encode=lambda text:tok.encode(text,add_special_tokens=False)
    for r in inputs:
        baseline=base[r['id']];target=targets[r['id']]
        assert baseline['messages']==r['messages']
        assert baseline['is_gameable']==target['is_gameable']==r['is_gameable']
        messages=paired_messages(r['messages'],baseline,C['instruction'],r['memory'],r['user_query'])
        plain=render_context(tok,messages)
        # The system content contains only learned vectors; task text is in one user message.
        marked=[dict(m) for m in messages];marked[0]['content']+=MARKER
        rendered=render_context(tok,marked)
        assert rendered.count(MARKER)==1
        before,after=rendered.split(MARKER)
        assert before=='<|im_start|>system\n'
        if r is inputs[0]:write(OUT/'context_example.json',dict(id=r['id'],messages=messages,rendered_context=rendered.replace(MARKER,'[learned soft-prompt embeddings]'),target=final_text(target)))
        assert before+after==plain
        text=final_text(target)
        ids=encode(text)+[tok.eos_token_id]
        assert encode(plain+text)==encode(plain)+ids[:-1]
        ex=dict(id=r['id'],before=encode(before),after=encode(after),plain=encode(plain),target=ids,groups=[1]*len(ids))
        assert len(ex['before'])+max(C['lengths'])+len(ex['after'])+len(ids)<=C['max_context'],r['id']
        split='valid' if json.dumps(r['messages'],sort_keys=True) in held else 'train'
        all_data[split].append(ex)
    rng=random.Random(C['seed']+2)
    all_data['diagnostic_indices']=rng.sample(range(len(all_data['train'])),C['train_eval_count'])
    audit={split:dict(n=len(ex),target_tokens=sum(len(e['target']) for e in ex),mean_context_tokens=sum(len(e['plain']) for e in ex)/len(ex),max_total_tokens=max(len(e['plain'])+max(C['lengths'])+len(e['target']) for e in ex)) for split,ex in all_data.items() if split!='diagnostic_indices'}
    write(OUT/'dataset.json',dict(sources=C['sources'],audit=audit,split_ids={s:[e['id'] for e in all_data[s]] for s in ['train','valid']},diagnostic_ids=[all_data['train'][i]['id'] for i in all_data['diagnostic_indices']],reasoning=False,test_used=False))
    save_torch(DATA/'prepared.pt',all_data)
    return all_data

def load_model(path,revision=None):
    model=Qwen3_5ForCausalLM.from_pretrained(path,revision=revision,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    model.config.use_cache=False
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    assert q.chunk_gated_delta_rule is not None, 'FLA kernel missing'
    for p in model.parameters():p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model

def nll(model,e,z=None,backward=False,denom=1):
    E=model.get_input_embeddings();device=E.weight.device
    embed=lambda ids:E(torch.tensor(ids,device=device))
    if z is None:
        prefix=embed(e['plain'])
    else:prefix=torch.cat([embed(e['before']),z.to(E.weight.dtype),embed(e['after'])])
    x=torch.cat([prefix,embed(e['target'])])[None]
    h=model.model(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device=device,dtype=torch.long),use_cache=False).last_hidden_state[0,len(prefix)-1:-1]
    y=torch.tensor(e['target'],device=device)
    def ce(h,y):return F.cross_entropy(model.lm_head(h).float(),y,reduction='none')
    losses=[]
    for i in range(0,len(y),128):
        # Retain only scalar losses in the backward graph; checkpoint the vocabulary projection.
        if backward:
            def summed(h,y):return ce(h,y).sum()
            losses.append(checkpoint(summed,h[i:i+128],y[i:i+128],use_reentrant=False))
        else:losses.append(ce(h[i:i+128],y[i:i+128]))
    if backward:
        loss=torch.stack(losses).sum()/denom;loss.backward();return float(loss.detach())
    return torch.cat(losses).double()

@torch.no_grad()
def evaluate(model,examples,z,label):
    model.eval();sums=[0.,0.,0.];counts=[0,0,0];per=[]
    for i,e in enumerate(examples):
        v=nll(model,e,z);groups=torch.tensor(e['groups'],device=v.device)
        for k in range(3):
            chosen=v[groups==k];sums[k]+=float(chosen.sum());counts[k]+=len(chosen)
        per.append(dict(id=e['id'],nll=float(v.mean()),tokens=len(v)))
        if i%16==0:progress('evaluating',label=label,completed=i,total=len(examples))
    return dict(nll=sum(sums)/sum(counts),tokens=sum(counts),examples=len(examples),**{name:dict(nll=sums[k]/counts[k] if counts[k] else None,tokens=counts[k]) for k,name in enumerate(NAMES)}),per

def candidate(model,k):
    path=OUT/f'k{k}'/'latest.pt'
    if path.exists():
        saved=torch.load(path,weights_only=False,map_location='cpu')
        z=saved['z'].to('cuda').requires_grad_(True)
        opt=torch.optim.Adam([z],lr=C['learning_rate'],weight_decay=C['weight_decay']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        return dict(z=z,opt=opt,state=saved['state'])
    # Shared first 128 vectors; initialization is independent of minibatch order.
    g=torch.Generator(device='cuda').manual_seed(C['seed'])
    z=(torch.randn(256,model.config.hidden_size,device='cuda',dtype=torch.float32,generator=g)*model.get_input_embeddings().weight[:8192].float().std()).detach()[:k].clone().requires_grad_(True)
    return dict(z=z,opt=torch.optim.Adam([z],lr=C['learning_rate'],weight_decay=C['weight_decay']),state=dict(step=0,rate=C['learning_rate'],bad_evals=0,stopped=False))

def save(k,c):
    save_torch(OUT/f'k{k}'/'latest.pt',dict(z=c['z'].detach().cpu(),optimizer=c['opt'].state_dict(),state=c['state'],cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),source_commit=COMMIT))

def block(model,k,c,data,order,end):
    model.train();s=c['state'];z=c['z'];opt=c['opt']
    while s['step']<end:
        step=s['step'];batch=[data['train'][j] for j in order[step*C['batch_size']:(step+1)*C['batch_size']]]
        rate=s['rate']*min(1.,(step+1)/C['warmup_steps'])
        for group in opt.param_groups:group['lr']=rate
        progress('training',length=k,step=step+1,lr=rate)
        start=time.time();opt.zero_grad(set_to_none=True);tokens=sum(len(e['target']) for e in batch)
        before=z.detach().clone();loss=sum(nll(model,e,z,True,tokens) for e in batch)
        assert z.grad is not None and torch.isfinite(z.grad).all() and torch.isfinite(torch.tensor(loss))
        norm=float(torch.nn.utils.clip_grad_norm_([z],C['clip_norm']));opt.step()
        delta=float((z.detach()-before).norm());assert delta>0 and torch.isfinite(z).all()
        s['step']+=1
        if s['step']%10==0:save(k,c)
        append(OUT/f'k{k}'/'training.jsonl',dict(step=s['step'],nll=loss,lr=rate,grad_norm=norm,update_norm=delta,tokens=tokens,processed_tokens=sum(len(e['plain'])+k+len(e['target']) for e in batch),examples=len(batch),seconds=time.time()-start,time=time.time(),peak_bytes=torch.cuda.max_memory_allocated()))
    save(k,c)

def eval_candidate(model,k,c,data,hist):
    s=c['state'];d=OUT/f'k{k}';d.mkdir(exist_ok=True)
    train,train_rows=evaluate(model,[data['train'][i] for i in data['diagnostic_indices']],c['z'],f'k{k} train step {s["step"]}')
    valid,valid_rows=evaluate(model,data['valid'],c['z'],f'k{k} validation step {s["step"]}')
    old_best=s.get('best',float('inf'))
    action=plateau(s,valid['nll'],C)
    record=dict(step=s['step'],train=train,valid=valid,next_lr=s['rate'],action=action,state=dict(s),time=time.time())
    append(d/'evaluations.jsonl',record);hist.append(record)
    write(d/f'eval_{s["step"]:06d}.json',dict(summary=record,train=train_rows,valid=valid_rows))
    if valid['nll']<old_best:save_torch(d/'best.pt',dict(z=c['z'].detach().cpu(),step=s['step'],valid_nll=valid['nll'],source_commit=COMMIT))
    save(k,c)
    return record

def main():
    global COMMIT
    OUT.mkdir(exist_ok=True)
    provenance=json.loads((HERE/'source_manifest.json').read_text())
    COMMIT=provenance['source_commit']
    for name,sha in provenance['files'].items():assert digest(HERE/name)==sha,name
    import importlib.metadata
    assert importlib.metadata.version('flash-linear-attention')=='0.5.2'
    assert importlib.metadata.version('fla-core')=='0.5.2'
    import triton
    assert triton.__version__=='3.7.1'
    startfile=OUT/'manifest.json'
    if startfile.exists():
        manifest=json.loads(startfile.read_text());assert manifest['source_commit']==COMMIT
    else:
        import transformers
        manifest=dict(source_commit=COMMIT,config=C,start_time=time.time(),torch=torch.__version__,transformers=transformers.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,parameter_dtype='float32 soft prompt and Adam state; bf16 model and forward embeddings',scope='Final-answer paired translation; soft prompt only; no reasoning, verbalization, generation, or behavior evaluation')
        write(startfile,manifest)
    deadline=manifest['start_time']+C['max_wall_hours']*3600
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'],local_files_only=True)
    data=prepare(tok)
    diagnostics=[data['train'][i] for i in data['diagnostic_indices']]
    basepath=OUT/'baselines.json';baselines=json.loads(basepath.read_text()) if basepath.exists() else {}
    progress('loading_base');model=load_model(C['model'],C['revision'])
    from test_objective import validate
    write(OUT/'objective_check.json',validate(model,tok,data))
    if 'base' not in baselines:
        result={}
        for split,ex in [('train',diagnostics),('valid',data['valid'])]:
            summary,per=evaluate(model,ex,None,'base '+split);result[split]=summary;write(OUT/f'base_{split}.json',per)
        baselines['base']=result;write(basepath,baselines)
    g=torch.Generator().manual_seed(C['seed']+1);order=[]
    while len(order)<C['max_steps_per_candidate']*C['batch_size']:order+=torch.randperm(len(data['train']),generator=g).tolist()
    write(OUT/'batch_order.json',dict(seed=C['seed']+1,sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest()))
    candidates={};hist={}
    for k in C['lengths']:
        (OUT/f'k{k}').mkdir(exist_ok=True);candidates[k]=candidate(model,k)
        p=OUT/f'k{k}'/'evaluations.jsonl';hist[k]=rows(p) if p.exists() else []
        state=candidates[k]['state']
        hist[k]=[r for r in hist[k] if r['step']<=state['step']]
        if hist[k] and hist[k][-1]['step']==state['step'] and 'state' in hist[k][-1]:
            candidates[k]['state']=hist[k][-1]['state']
        # Remove uncheckpointed metrics after an interrupted block, avoiding duplicate steps.
        for filename in ['training.jsonl','evaluations.jsonl']:
            path=OUT/f'k{k}'/filename
            if path.exists():
                kept=[r for r in rows(path) if r['step']<=state['step']]
                path.write_text(''.join(json.dumps(r)+'\n' for r in kept))
        # A crash after checkpointing training but before evaluation must evaluate that step.
        if not hist[k] or hist[k][-1]['step']<candidates[k]['state']['step']:eval_candidate(model,k,candidates[k],data,hist[k])
    selection_path=OUT/'selection.json';selection=json.loads(selection_path.read_text()) if selection_path.exists() else None
    target=C['comparison_steps'];budget=False
    if selection is None:
        while True:
            for end in range(C['eval_every'],target+1,C['eval_every']):
                for k,c in candidates.items():
                    if c['state']['step']>=end or c['state']['stopped']:continue
                    if time.time()>=deadline:budget=True;break
                    block(model,k,c,data,order,end);eval_candidate(model,k,c,data,hist[k])
                if budget:break
            if budget:break
            decision=choose(hist,C,allow_extend=target<C['extended_comparison_steps'])
            append(OUT/'selection_history.jsonl',dict(step=target,**decision))
            if decision['extend']:
                target=C['extended_comparison_steps'];continue
            selection=dict(**decision,comparison_steps=target,time=time.time());write(selection_path,selection);break
    if not budget:
        k=selection['selected'];c=candidates[k]
        while c['state']['step']<C['max_steps_per_candidate'] and not c['state']['stopped']:
            if time.time()>=deadline:budget=True;break
            end=min(c['state']['step']+C['eval_every'],C['max_steps_per_candidate'])
            block(model,k,c,data,order,end);eval_candidate(model,k,c,data,hist[k])
        reason='wall_time_budget' if budget else ('validation_plateau' if c['state']['stopped'] else 'step_budget')
    else:reason='wall_time_budget'
    write(OUT/'outcome.json',dict(status='complete',reason=reason,selection=selection,states={k:c['state'] for k,c in candidates.items()},finished=time.time(),source_commit=COMMIT))
    progress('complete',reason=reason)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(OUT/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()))
        progress('failed',error=str(e));raise

