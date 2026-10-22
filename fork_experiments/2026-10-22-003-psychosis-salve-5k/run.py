"""Soft-prompt likelihood experiment only. No generation, search, or judging."""
import os,json,time,hashlib,subprocess,shutil,random,traceback,re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM,AutoTokenizer
from policy import plateau,choose
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
C=json.loads((HERE/'config.json').read_text());OUT=HERE/'results';DATA=Path(C['resume_from'])/'assets/dataset'
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
    for rel,expected in C['source_hashes'].items():
        dst=DATA/rel
        if not dst.exists():
            src=Path(C['source_root'])/'results'/rel;assert digest(src)==expected,rel
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        assert digest(dst)==expected,rel
    cache=DATA/'prepared.pt'
    if cache.exists():return torch.load(cache,weights_only=False)
    all_data={};audit={};systems=set();prompt_sets={}
    for split,count in [('train',10000),('valid',100)]:
        inputs=rows(DATA/'inputs'/f'{split}.jsonl');assert len(inputs)==count
        targets={r['id']:r for r in rows(DATA/'generations'/f'teacher_{split}.jsonl')}
        p=DATA/'generations'/f'teacher_{split}_retry.jsonl'
        if p.exists():targets.update({r['id']:r for r in rows(p)})
        assert len(targets)==count and len({r['id'] for r in inputs})==count
        prompt_sets[split]={json.dumps(r['messages'],sort_keys=True) for r in inputs}
        examples=[]
        for r in inputs:
            target=targets[r['id']];text=target['completion'];assert target['accepted'] and target['finish_reason']=='stop'
            assert text.count('</think>')==1 and target['is_gameable']==r['is_gameable']
            systems.add(r['messages'][0]['content'])
            marker='SALVE_SOFT_PROMPT_PLACEHOLDER'
            messages=[dict(x) for x in r['messages']];messages[0]['content']+='\n\n'+marker
            rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            assert rendered.count(marker)==1 and rendered.endswith('<|im_start|>assistant\n')
            before,after=rendered.split(marker);encode=lambda s:tok.encode(s,add_special_tokens=False)
            plain=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            ids=enc['input_ids']+[tok.eos_token_id]
            assert encode(plain+text)==encode(plain)+ids[:-1]
            think_end=text.index('</think>')+len('</think>')
            m=re.search(r'\[\[\s*##\s*response\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)',text[think_end:],re.S)
            assert m and m.group(1).strip()==target['answer']
            a=think_end+m.start(1);b=think_end+m.end(1)
            while a<b and text[a].isspace():a+=1
            while b>a and text[b-1].isspace():b-=1
            groups=[0 if end<=think_end else (1 if end>a and start<b else 2) for start,end in enc['offset_mapping']]+[2]
            ex=dict(id=r['id'],before=encode(before),after=encode(after),plain=encode(plain),target=ids,groups=groups)
            assert len(ex['before'])+256+len(ex['after'])+len(ids)<=C['max_context']
            assert 0 in groups and 1 in groups
            examples.append(ex)
        all_data[split]=examples
        audit[split]=dict(n=len(examples),tokens=sum(len(e['target']) for e in examples),max_response=max(len(e['target']) for e in examples))
    assert len(systems)==1 and not prompt_sets['train']&prompt_sets['valid']
    g=torch.Generator().manual_seed(C['seed'])
    all_data['diagnostic_indices']=torch.randperm(10000,generator=g)[:C['train_eval_count']].tolist()
    save_torch(cache,all_data)
    write(OUT/'dataset.json',dict(source_commit=C['data_source_commit'],sha256=C['source_hashes'],audit=audit,diagnostic_ids=[all_data['train'][i]['id'] for i in all_data['diagnostic_indices']],test_used=False))
    return all_data

def load_model(path,revision=None):
    model=AutoModelForCausalLM.from_pretrained(path,revision=revision,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    model.config.use_cache=False
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
    return dict(nll=sum(sums)/sum(counts),tokens=sum(counts),examples=len(examples),**{name:dict(nll=sums[k]/counts[k],tokens=counts[k]) for k,name in enumerate(NAMES)}),per

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
    z=(torch.randn(256,model.config.hidden_size,device='cuda',dtype=torch.float32,generator=g)*model.get_input_embeddings().weight.float().std()).detach()[:k].clone().requires_grad_(True)
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
        append(OUT/f'k{k}'/'training.jsonl',dict(step=s['step'],nll=loss,lr=rate,grad_norm=norm,update_norm=delta,tokens=tokens,seconds=time.time()-start,time=time.time(),peak_bytes=torch.cuda.max_memory_allocated()))
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


def initialize_resume():
    parent=Path(C['resume_from']);checkpoint=parent/'results/k256/latest.pt'
    assert digest(checkpoint)==C['resume_checkpoint_sha256'],'Resume checkpoint changed'
    if (OUT/'resume.json').exists():return
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['source_commit']==C['resume_training_commit']
    assert saved['state']['step']==C['resume_step'] and not saved['state']['stopped']
    assert saved['z'].dtype==torch.float32 and tuple(saved['z'].shape)==(256,2048)
    assert all(k in saved for k in ['optimizer','cpu_rng','cuda_rng','python_rng','numpy_rng'])
    assert saved['optimizer']['state'] and all(int(v['step'])==C['resume_step'] for v in saved['optimizer']['state'].values())
    last=rows(parent/'results/k256/evaluations.jsonl')[-1]
    assert last['step']==C['resume_step'] and last['state']==saved['state']
    (HERE/'assets').mkdir(exist_ok=True)
    shutil.copyfile(checkpoint,HERE/'assets/resume_latest.pt')
    for name in ['baselines.json','dataset.json','selection.json','selection_history.jsonl','base_train.json','base_valid.json','teacher_train.json','teacher_valid.json']:
        src=parent/'results'/name
        if src.exists():shutil.copyfile(src,OUT/name)
    for k in [128,256]:
        dst=OUT/f'k{k}';dst.mkdir(exist_ok=True)
        for src in (parent/'results'/f'k{k}').iterdir():
            if src.suffix in ['.json','.jsonl'] or k==256 and src.name in ['latest.pt','best.pt']:shutil.copyfile(src,dst/src.name)
    assert (DATA/'prepared.pt').is_file()
    write(OUT/'resume.json',dict(parent_experiment=str(parent),parent_commit=saved['source_commit'],checkpoint_sha256=digest(checkpoint),prepared_cache_sha256=digest(DATA/'prepared.pt'),start_step=saved['state']['step'],state=saved['state'],optimizer_steps=[int(v['step']) for v in saved['optimizer']['state'].values()],old_batch_order=json.loads((parent/'results/batch_order.json').read_text()),time=time.time()))

def main():
    global COMMIT
    OUT.mkdir(exist_ok=True)
    COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    relevant=[str(HERE.relative_to(ROOT)),'salve']
    assert not subprocess.check_output(['git','diff','HEAD','--',*relevant],cwd=ROOT), 'Uncommitted experiment code'
    assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',str(HERE.relative_to(ROOT))],cwd=ROOT), 'Untracked experiment code'
    initialize_resume()
    startfile=OUT/'manifest.json'
    if startfile.exists():
        manifest=json.loads(startfile.read_text());assert manifest['source_commit']==COMMIT
    else:
        import transformers
        manifest=dict(source_commit=COMMIT,config=C,start_time=time.time(),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),parameter_dtype='float32 soft prompt and Adam state; bf16 model and forward embeddings',scope='Likelihood only; no verbalization or behavior evaluation')
        write(startfile,manifest)
    deadline=manifest['start_time']+C['max_wall_hours']*3600
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'],local_files_only=True)
    data=prepare(tok)
    diagnostics=[data['train'][i] for i in data['diagnostic_indices']]
    basepath=OUT/'baselines.json';baselines=json.loads(basepath.read_text()) if basepath.exists() else {}
    if 'teacher' not in baselines:
        progress('loading_teacher')
        teacher=load_model(str(Path(C['source_root'])/'assets/teacher-merged'))
        teacher_result={}
        for split,ex in [('train',diagnostics),('valid',data['valid'])]:
            summary,per=evaluate(teacher,ex,None,'teacher '+split);teacher_result[split]=summary;write(OUT/f'teacher_{split}.json',per)
        baselines['teacher']=teacher_result;write(basepath,baselines)
        del teacher;import gc;gc.collect();torch.cuda.empty_cache()
    progress('loading_base');model=load_model(C['model'],C['revision'])
    if 'base' not in baselines:
        result={}
        for split,ex in [('train',diagnostics),('valid',data['valid'])]:
            summary,per=evaluate(model,ex,None,'base '+split);result[split]=summary;write(OUT/f'base_{split}.json',per)
        baselines['base']=result;write(basepath,baselines)
    g=torch.Generator().manual_seed(C['seed']+1);order=[]
    while len(order)<C['max_steps_per_candidate']*C['batch_size']:order+=torch.randperm(len(data['train']),generator=g).tolist()
    old_order=order[:C['resume_step']*C['batch_size']]
    old_fingerprint=json.loads((OUT/'resume.json').read_text())['old_batch_order']['sha256']
    assert hashlib.sha256(json.dumps(old_order).encode()).hexdigest()==old_fingerprint,'Batch sequence does not extend the original run'
    write(OUT/'batch_order.json',dict(seed=C['seed']+1,sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),original_prefix_sha256=old_fingerprint,next_batch=order[C['resume_step']*C['batch_size']:(C['resume_step']+1)*C['batch_size']]))
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
    if not (OUT/'resume_validation.json').exists():
        reference=hist[256][-1]['valid']['nll']
        observed,per=evaluate(model,data['valid'],candidates[256]['z'],'resume verification step 2500')
        delta=observed['nll']-reference
        write(OUT/'resume_validation.json',dict(step=2500,reference_nll=reference,resumed_nll=observed['nll'],delta=delta,tolerance=.002,passed=abs(delta)<.002,validation=observed,per_example=per))
        assert abs(delta)<.002,'Resumed validation differs unexpectedly from saved run'
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
