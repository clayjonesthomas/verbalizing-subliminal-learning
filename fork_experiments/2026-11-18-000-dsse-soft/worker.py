"""DSSE fixed data; fast batch pilot, selection-only LR sweep, winner readout."""
import argparse, functools, hashlib, json, lzma, math, os, random, statistics, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from salve.objectives import NLLObjective
from salve.soft_prompt import make_soft_prompt_input, embed_soft_prompt_inputs, sample_soft_prompt_batch
from salve.salve import SearchConfig, VERBALIZE_PROMPTS, system_header, stop_token_ids
from salve.bucket_search import bucket_search, BucketConfig
from podrun.activation_checkpointing import Policy
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
RESUME=None

def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def runtime():
    global ART,SHARED,RUN
    ART=Path(os.environ['PODRUN_ARTIFACT_DIR']);ART.mkdir(parents=True,exist_ok=True)
    RUN=os.environ['PODRUN_ID'];SHARED=Path('/workspace/projects/verbalizing-subliminal-learning/experiments')/HERE.name/'shared';SHARED.mkdir(parents=True,exist_ok=True)

def progress(**kw):
    atomic(Path(os.environ['PODRUN_DIR'])/'progress.json',dict(time=time.time(),**kw));print(json.dumps(kw),flush=True)

def publish(stage,obj):
    result=dict(stage=stage,run_id=RUN,source_commit=os.environ['PODRUN_COMMIT'],artifact_root=str(ART),**obj)
    atomic(HERE/'measurements'/f'{RUN}-{stage}.json',result);return result

def seed():
    random.seed(C['seed']);torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])

def load():
    seed();t=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['model_revision']);tok.pad_token=tok.eos_token
    tok.apply_chat_template=functools.partial(tok.apply_chat_template,enable_thinking=False)
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['model_revision'],dtype=torch.bfloat16).cuda().eval();model.requires_grad_(False)
    policy=Policy.from_environment();policy.configure_model(model);policy.assert_model(model)
    assert not policy.backbone and not policy.head
    progress(stage='loaded',seconds=time.perf_counter()-t)
    return NLLObjective(model,tok)

def initial(obj):
    seed();e=obj.model.get_input_embeddings().weight
    return (torch.randn(C['soft_tokens'],e.shape[1],device=e.device,dtype=e.dtype)*e.std()).requires_grad_(True)

def split_indices(n):
    ids=list(range(n));random.Random(C['seed']).shuffle(ids)
    return ids[C['selection_rows']:],ids[:C['selection_rows']]

def prepare(obj):
    from huggingface_hub import hf_hub_download
    args=dict(repo_id='Transluce/DSSE',repo_type='dataset',revision=C['dataset_revision'])
    file=hf_hub_download(filename='datasets/'+C['behavior']+'.jsonl.xz',**args)
    with lzma.open(file,'rt') as f:
        d=next(d for line in f if (d:=json.loads(line))['dataset_id']==C['dataset_id'])
    bfile=hf_hub_download(filename='behaviors.json',**args)
    behavior=next(b for b in json.loads(Path(bfile).read_text()) if b['behavior_id']==C['behavior'])
    assert len(d['rows'])==148 and len(behavior['eval_prompts'])==100
    # Include response EOS identically in soft and student NLL. This dataset's longest row is <1024.
    pairs=[(r['prompt'],obj.tokenizer.encode(r['response'],add_special_tokens=False)+[obj.tokenizer.eos_token_id]) for r in d['rows']]
    tr,va=split_indices(len(pairs))
    prepared=obj.prepare(pairs)
    atomic(ART/'data.json',dict(dataset=d,behavior=behavior,train_indices=tr,selection_indices=va,revision=C['dataset_revision'],compressed_sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest()))
    return [prepared[i] for i in tr],[prepared[i] for i in va],prepared,behavior,d

def timed(fn):
    torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();return v,time.perf_counter()-t

def plateau(history):
    points=[r for r in history if r['step']%500==0]
    if not points or points[-1]['step']<C['min_steps'] or len(points)<4:return False
    best=[];v=float('inf')
    for p in points:v=min(v,p['selection_nll']);best.append(v)
    return all((a-b)/max(abs(a),1e-12)<C['plateau_relative'] for a,b in zip(best[-4:-1],best[-3:]))

def pilot(obj):
    train,val,_,_,_=prepare(obj);z=initial(obj);start=time.perf_counter();records=[];choices={}
    panel=sorted(train,key=lambda x:len(x.response_ids)+len(x.soft_input.after_prompt),reverse=True)[:8]
    base_grad=None;base_loss=None
    for kind,sizes in [('training',[8,16,32,64,128,256,512]),('scoring',[1,2,4,8,16,30]),('generation',[8,16,32,64])]:
        group=[]
        for size in sizes:
            if time.perf_counter()-start>C['pilot_seconds'] and group:break
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            try:
                if kind=='training':
                    examples=(panel*math.ceil(size/len(panel)))[:size]
                    def fn():
                        z.grad=None;return obj.loss(z,examples,backward=True,mini_batch_size=size)
                elif kind=='scoring':
                    def fn():return obj.loss_of_text('You are a careful and concise assistant. '*12,val,size)
                else:
                    inputs=[make_soft_prompt_input(obj.tokenizer,VERBALIZE_PROMPTS[i%len(VERBALIZE_PROMPTS)]['user'],prefill=VERBALIZE_PROMPTS[i%len(VERBALIZE_PROMPTS)]['prefill']+system_header(obj.tokenizer)+'I am a helpful assistant. '*15) for i in range(size)]
                    def fn():return sample_soft_prompt_batch(obj.model,inputs,z.detach(),16,.7,stop_token_ids(obj.model,obj.tokenizer),use_cache=True)
                fn();values=[]
                for _ in range(2):v,secs=timed(fn);values.append(secs)
                secs=statistics.median(values)
                work=size if kind=='training' else (30 if kind=='scoring' else sum(len(x) for x in v))
                row=dict(kind=kind,batch=size,ok=True,seconds=secs,throughput=work/secs,peak_gib=torch.cuda.max_memory_allocated()/2**30)
                if kind=='training':
                    g=z.grad.detach().float().clone()
                    if base_grad is None:base_grad=g;base_loss=v
                    row.update(loss=v,loss_difference=v-base_loss,gradient_relative=float((g-base_grad).norm()/base_grad.norm()),gradient_cosine=float(torch.nn.functional.cosine_similarity(g.flatten(),base_grad.flatten(),dim=0)))
                group.append(row);records.append(row);progress(stage='pilot',**row)
                # Confirm decline with both timed repetitions, not one noisy sample.
                if len(group)>1 and all(work/s < group[-2]['throughput']*.98 for s in values):break
            except torch.cuda.OutOfMemoryError:
                records.append(dict(kind=kind,batch=size,ok=False,oom=True));z.grad=None;torch.cuda.empty_cache();break
        if not group:raise RuntimeError('No successful '+kind+' benchmark')
        choices[kind]=max(group,key=lambda r:r['throughput'])['batch']
    z.grad=None
    result=publish('pilot',dict(choices=choices,records=records,seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),scope='warmed benchmark; excludes loading; generation sizes above64 cannot be used by64-attempt rounds'))
    atomic(SHARED/'pilot.json',result)

def fit(obj,lr):
    train,val,_,_,_=prepare(obj);decision=json.loads((SHARED/'pilot.json').read_text());mb=decision['choices']['training'];score_mb=decision['choices']['scoring']
    z=initial(obj);opt=torch.optim.Adam([z],lr=lr,weight_decay=.001);queue=[];history=[];checkpoints=[];train_seconds=0.;selection_seconds=0.
    def save(step):
        path=ART/f'checkpoint-{step:06d}.pt';torch.save(dict(z=z.detach().cpu(),optimizer=opt.state_dict(),step=step,queue=queue,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history,lr=lr,microbatch=mb),path)
        checkpoints.append(dict(step=step,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    nll,sec=timed(lambda:obj.loss(z,val,mini_batch_size=score_mb));selection_seconds+=sec
    history.append(dict(step=0,selection_nll=nll,training_seconds=0));save(0)
    for step in range(1,C['max_steps']+1):
        while len(queue)<mb:queue.extend(torch.randperm(len(train)).tolist())
        ids,queue=queue[:mb],queue[mb:]
        def update():
            opt.zero_grad();loss=obj.loss(z,[train[i] for i in ids],backward=True,mini_batch_size=mb);norm=float(torch.nn.utils.clip_grad_norm_([z],1.));opt.step();return loss,norm
        (loss,norm),secs=timed(update);train_seconds+=secs
        if step%C['eval_every']==0:
            nll,secs=timed(lambda:obj.loss(z,val,mini_batch_size=score_mb));selection_seconds+=secs
            history.append(dict(step=step,selection_nll=nll,train_nll=loss,gradient_norm=norm,training_seconds=train_seconds));progress(stage='sweep',lr=lr,microbatch=mb,**history[-1]);atomic(ART/'curve.json',history)
        if step%C['checkpoint_every']==0:
            save(step)
            if plateau(history):break
    best=min(history,key=lambda r:r['selection_nll'])
    result=publish('sweep',dict(lr=lr,history=history,checkpoints=checkpoints,best=best,training_seconds=train_seconds,selection_seconds=selection_seconds,microbatch=mb,effective_batch=mb,stop_reason='selection plateau' if plateau(history) else '10000-step ceiling'))
    atomic(SHARED/f'sweep-{lr:g}.json',result)

@torch.no_grad()
def answers(obj,behavior,batch,z=None,text=None):
    tok=obj.tokenizer;outs=[];model=obj.model;seed()
    for offset in range(0,len(behavior['eval_prompts']),batch):
        rows=behavior['eval_prompts'][offset:offset+batch];prompts=[r['prompt'] for r in rows]
        if z is not None:
            inputs=[make_soft_prompt_input(tok,q) for q in prompts];emb,mask,_=embed_soft_prompt_inputs(model,inputs,z)
            # Convert right-padded embedding sequences to left-padding for generate.
            left=torch.zeros_like(emb);lm=torch.zeros_like(mask)
            for i,n in enumerate(mask.sum(1).tolist()):left[i,-n:]=emb[i,:n];lm[i,-n:]=1
            kw=dict(inputs_embeds=left,attention_mask=lm);prefix=0
        else:
            tok.padding_side='left';texts=[tok.apply_chat_template(([dict(role='system',content=text)] if text is not None else [])+[dict(role='user',content=q)],tokenize=False,add_generation_prompt=True) for q in prompts]
            kw=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to(model.device);prefix=kw['input_ids'].shape[1]
        ids=model.generate(**kw,do_sample=True,temperature=.6,top_p=1.,top_k=0,max_new_tokens=C['max_new_tokens'],pad_token_id=tok.eos_token_id,use_cache=True)
        for q,seq in zip(prompts,ids[:,prefix:]):outs.append(dict(prompt=q,response=tok.decode(seq,skip_special_tokens=True)))
    assert len(outs)==100
    return outs

def evaluate(obj,behavior,name,z=None,text=None):
    from judging import judge_answers
    # Conservative independent behavior batch, not the short-search pilot batch.
    previous=RESUME/(name+'-answers.json') if RESUME else None
    if previous and previous.exists():
        out=json.loads(previous.read_text());secs=None
        assert [r['prompt'] for r in out]==[r['prompt'] for r in behavior['eval_prompts']]
        atomic(ART/(name+'-reuse.json'),dict(source=str(previous),sha256=hashlib.sha256(previous.read_bytes()).hexdigest()))
    else:out,secs=timed(lambda:answers(obj,behavior,8,z=z,text=text))
    atomic(ART/(name+'-answers.json'),out)
    t=time.perf_counter();scores,usage=judge_answers(behavior,out,ART/(name+'-judges.json'),config=C,shared=SHARED)
    return dict(score=statistics.mean(scores),generation_seconds=secs,judging_seconds=time.perf_counter()-t,usage=usage,n=100)

def readout(obj):
    _,val,_,behavior,_=prepare(obj)
    sweeps=[json.loads((SHARED/f'sweep-{lr:g}.json').read_text()) for lr in C['learning_rates']]
    winner=min(sweeps,key=lambda r:(r['best']['selection_nll'],r['lr']));atomic(ART/'winner.json',winner)
    decision=json.loads((SHARED/'pilot.json').read_text())['choices'];base=evaluate(obj,behavior,'base');base['selection_nll']=obj.loss_of_text(None,val,decision['scoring']);publish('baseline',base)
    for cp in winner['checkpoints']:
        path=Path(cp['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['sha256']
        z=torch.load(path,map_location='cuda',weights_only=False)['z'].detach();step=cp['step']
        cfg=SearchConfig(n_select=30,mini_batch_size=decision['scoring'],use_cache=True)
        t=time.perf_counter();search=bucket_search(obj,z,val,cfg,C['seed'],config=BucketConfig(rounds=6),generation_batch_size=decision['generation']);seconds=time.perf_counter()-t
        atomic(ART/f'search-{step}.json',search)
        if search['best'] is None:raise RuntimeError('No valid verbalization at step '+str(step))
        soft=evaluate(obj,behavior,f'soft-{step}',z=z);soft['selection_nll']=obj.loss(z,val,mini_batch_size=decision['scoring'])
        hard=evaluate(obj,behavior,f'text-{step}',text=search['best']['text']);hard['selection_nll']=search['best']['score']
        publish(f'readout-{step}',dict(lr=winner['lr'],step=step,soft=soft,verbalized=hard,text=search['best']['text'],search_seconds=seconds))
        progress(stage='readout',step=step,lr=winner['lr'],soft_score=soft['score'],text_score=hard['score'])

def student(obj):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig,SFTTrainer
    _,val,_,behavior,d=prepare(obj)
    if RESUME:
        from peft import PeftModel
        import ast,re
        source=json.loads((RESUME/'data.json').read_text());current=json.loads((ART/'data.json').read_text())
        assert source==current, 'Recovery dataset/split provenance mismatch'
        state=json.loads((RESUME/'student/checkpoint-74/trainer_state.json').read_text())
        assert state['global_step']==74 and state['epoch']==2
        adapter=RESUME/'adapter';cfg=json.loads((adapter/'adapter_config.json').read_text())
        assert cfg['r']==16 and cfg['lora_alpha']==32 and cfg['base_model_name_or_path']==C['model']
        provenance={str(f.relative_to(RESUME)):hashlib.sha256(f.read_bytes()).hexdigest() for f in adapter.iterdir() if f.is_file()}
        atomic(ART/'student-reuse.json',dict(source=str(RESUME),hashes=provenance,completed_steps=74))
        obj.model=PeftModel.from_pretrained(obj.model,str(adapter),is_trainable=False).eval();obj.model.requires_grad_(False)
        log=(RESUME.parent/'run.log').read_text();metrics=[ast.literal_eval(line) for line in log.splitlines() if line.startswith("{'train_runtime':")]
        assert len(metrics)==1
        result=evaluate(obj,behavior,'lora');result.update(training_seconds=float(metrics[0]['train_runtime']),training_seconds_scope='original completed training from TRL log',selection_nll=obj.loss_of_text(None,val,8),nll_scope='in-sample for LoRA, same30 selection rows as soft prompts',rows=148,reused_artifacts=str(RESUME))
        publish('student',result);return
    # TRL prompt/completion format gives response-only loss; fixed paper recipe.
    rows=[dict(prompt=[dict(role='user',content=r['prompt'])],completion=[dict(role='assistant',content=r['response'])]) for r in d['rows']]
    args=SFTConfig(output_dir=str(ART/'student'),learning_rate=.0005,num_train_epochs=2,per_device_train_batch_size=4,gradient_accumulation_steps=1,max_length=1024,bf16=True,fp16=False,gradient_checkpointing=False,completion_only_loss=True,report_to='none',save_strategy='epoch',logging_steps=1,seed=C['seed'],data_seed=C['seed'])
    # The tokenizer's partial chat-template method pins thinking off.
    trainer=SFTTrainer(model=obj.model,args=args,train_dataset=Dataset.from_list(rows),processing_class=obj.tokenizer,peft_config=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules='all-linear',task_type='CAUSAL_LM'))
    t=time.perf_counter();trainer.train();seconds=time.perf_counter()-t;trainer.save_model(str(ART/'adapter'));obj.model=trainer.model.eval();obj.model.requires_grad_(False)
    result=evaluate(obj,behavior,'lora');result.update(training_seconds=seconds,selection_nll=obj.loss_of_text(None,val,8),nll_scope='in-sample for LoRA, same30 selection rows as soft prompts',rows=148)
    publish('student',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','fit','readout','student']);p.add_argument('--lr',type=float);p.add_argument('--resume-artifacts',type=Path);a=p.parse_args();RESUME=a.resume_artifacts;runtime();obj=load()
    if a.stage=='fit':fit(obj,a.lr)
    else:globals()[a.stage](obj)
