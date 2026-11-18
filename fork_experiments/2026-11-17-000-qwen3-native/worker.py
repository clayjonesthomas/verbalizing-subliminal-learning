"""Figure 4 training microbatch pilot and eight independent bucket recoveries."""
import functools, argparse, dataclasses, hashlib, json, math, os, random, statistics, time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from salve.objectives import NLLObjective
from salve.salve import SoftPromptConfig, SearchConfig, warmup_cosine
from salve.bucket_search import bucket_search, BucketConfig
from subliminal_learning.datasets import load_splits
from experiment_scripts.prompted_subliminal_learning.recover_and_evaluate import evaluate_prompt
from subliminal_learning.animal_prompts import trait_prompt
from podrun.activation_checkpointing import Policy

HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
ART=Path(os.environ['PODRUN_ARTIFACT_DIR']);ART.mkdir(parents=True,exist_ok=True)
RUN=os.environ['PODRUN_ID']
SHARED=Path('/workspace/projects/verbalizing-subliminal-learning/experiments')/HERE.name/'shared'
SHARED.mkdir(parents=True,exist_ok=True)

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def publish(value):
    value.update(run_id=RUN,source_commit=os.environ['PODRUN_COMMIT'])
    atomic(HERE/'measurements'/f'{RUN}.json',value)

def progress(**value):
    value['time']=time.time();atomic(Path(os.environ['PODRUN_DIR'])/'progress.json',value);print(json.dumps(value),flush=True)

def load():
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['model_revision'])
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    tok.apply_chat_template=functools.partial(tok.apply_chat_template,enable_thinking=C['enable_thinking'])
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['model_revision'],dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False);policy=Policy.from_environment();policy.configure_model(model);policy.assert_model(model)
    assert not policy.backbone and not policy.head
    return NLLObjective(model,tok)

def data(obj,animal):
    manifest=json.loads((SHARED/(animal+'-data.json')).read_text())
    assert manifest['model']==C['model'] and manifest['model_revision']==C['model_revision']
    assert manifest['enable_thinking']==C['enable_thinking']
    root=Path(manifest['path'])
    for name,sha in manifest['hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    splits=load_splits(root)
    assert len(splits['train'])==10000 and len(splits['validation'])==500
    atomic(ART/'input-manifest.json',manifest)
    return obj.prepare(splits['train']),obj.prepare(splits['validation'])

def initial(obj,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    e=obj.model.get_input_embeddings().weight
    return (torch.randn(128,e.shape[1],device=e.device,dtype=e.dtype)*e.std()).requires_grad_(True)

def run(obj,animal,seed,resume_checkpoint=None):
    decision={'selected_generation_batch':C['generation_batch']};mb=C['training_microbatch']
    train,val=data(obj,animal);cfg=SoftPromptConfig(mini_batch_size=mb);search=SearchConfig(use_cache=True)
    z=initial(obj,seed);steps=cfg.epochs*math.ceil(len(train)/cfg.train_batch_size)
    optimizer=torch.optim.Adam([z],lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,warmup_cosine(steps,round(cfg.warmup_frac*steps)))
    queue=[];history=[];start=time.perf_counter()
    start_step=0
    if resume_checkpoint:
        from training_resume import restore
        start_step,queue,history=restore(resume_checkpoint,z,optimizer,scheduler,steps,len(train))
        atomic(ART/'resume-provenance.json',dict(checkpoint=str(resume_checkpoint),sha256=hashlib.sha256(Path(resume_checkpoint).read_bytes()).hexdigest(),step=start_step,animal=animal,seed=seed))
    progress(stage='training_start',animal=animal,seed=seed,microbatch=mb,effective_batch=cfg.train_batch_size,step=start_step,total=steps)
    for step in range(start_step,steps):
        while len(queue)<cfg.train_batch_size:queue.extend(torch.randperm(len(train)).tolist())
        ids,queue=queue[:cfg.train_batch_size],queue[cfg.train_batch_size:]
        optimizer.zero_grad();loss=obj.loss(z,[train[i] for i in ids],backward=True,mini_batch_size=mb)
        norm=float(torch.nn.utils.clip_grad_norm_([z],1.));optimizer.step();scheduler.step()
        row=dict(step=step+1,loss=loss,unclipped_grad_norm=norm,learning_rate=optimizer.param_groups[0]['lr']);history.append(row)
        if (step+1)%50==0:progress(stage='training',animal=animal,seed=seed,total=steps,**row)
        if (step+1)%100==0 or step+1==steps:
            tmp=ART/'checkpoint.tmp';torch.save(dict(z=z.detach().cpu(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step+1,queue=queue,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history),tmp);tmp.replace(ART/'checkpoint.pt')
    training_seconds=time.perf_counter()-start;z=z.detach()
    soft_val=obj.loss(z,val,mini_batch_size=8)
    def on_round(nodes,stats):
        atomic(ART/'bucket-live.json',dict(nodes=nodes,stats=stats));progress(stage='bucket',animal=animal,seed=seed,round=stats['round'],candidates=len(nodes))
    start=time.perf_counter();result=bucket_search(obj,z,train,search,seed,config=BucketConfig(rounds=6),on_round=on_round,generation_batch_size=decision['selected_generation_batch'])
    search_seconds=time.perf_counter()-start;atomic(ART/'bucket-search.json',result)
    if result['best'] is None:raise RuntimeError('Bucket search yielded no valid candidate')
    prompts={'recovered':result['best']['text'],'teacher':trait_prompt(animal),'empty_prompt':'','default_prompt':None}
    evaluations={}
    for name,text in prompts.items():
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        ev=evaluate_prompt(obj,val,animal,text);answers=ev.pop('answers');atomic(ART/f'{name}-answers.json',answers)
        evaluations[name]=dict(text=text,**ev);progress(stage='evaluation',animal=animal,seed=seed,prompt=name,hit_rate=ev['hit_rate'])
    from soft_preference import evaluate as evaluate_soft_preference
    evaluate_soft_preference(obj,z,animal,seed,RUN,ART/'checkpoint.pt')
    publish(dict(stage='recovery',status='complete',animal=animal,seed=seed,soft_config=dataclasses.asdict(cfg),bucket_config=dataclasses.asdict(BucketConfig()),search_config=dataclasses.asdict(search),generation_batch_size=decision['selected_generation_batch'],training_seconds=training_seconds,training_seconds_scope="current_attempt" if start_step else "full_training",resumed_from_step=start_step,search_seconds=search_seconds,soft_val=soft_val,prompts=evaluations,best=result['best'],shortest=result['shortest'],rounds=result['rounds'],artifact_root=str(ART)))


def query_list(n):
    import numpy as np
    from subliminal_learning.teachers.prompted import PAPER_PROMPT_DIMENSIONS
    from subliminal_learning.vendored.divergence_tokens import PromptGenerator
    generator=PromptGenerator(rng=np.random.Generator(np.random.PCG64(C['generation_query_seed'])),**PAPER_PROMPT_DIMENSIONS)
    return [generator.sample_query() for _ in range(n)]


def number_pilot(obj):
    tok=obj.tokenizer;tok.padding_side='left'
    messages=[{'role':'system','content':trait_prompt('cat')},{'role':'user','content':query_list(1)[0]}]
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    one=tok([text],return_tensors='pt',add_special_tokens=False).to(obj.model.device)
    records=[];started=time.perf_counter();reason='candidate ceiling';last_seconds=0
    for size in C['pilot_batches']:
        if records and time.perf_counter()-started+max(1,last_seconds*2.5)>C['pilot_seconds']:
            reason='30-second time budget';break
        inputs={k:v.expand(size,-1).contiguous() for k,v in one.items()}
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.manual_seed(0);torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():
                outputs=obj.model.generate(**inputs,do_sample=True,temperature=C['generation_temperature'],max_new_tokens=C['pilot_tokens'],min_new_tokens=C['pilot_tokens'],use_cache=True,pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize();last_seconds=time.perf_counter()-t
            row=dict(batch=size,seconds=last_seconds,rows_per_second=size/last_seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30,ok=True);del outputs
        except torch.cuda.OutOfMemoryError:
            row=dict(batch=size,ok=False,oom=True);reason='OOM'
        del inputs;torch.cuda.empty_cache();records.append(row);progress(stage='number_generation_pilot',**row)
        if not row['ok']:break
        if len(records)>1 and row['rows_per_second']<records[-2]['rows_per_second']*.98:
            reason='throughput plateau';break
    valid=[r for r in records if r['ok']]
    if not valid:raise RuntimeError('Starting batch64 does not fit; no unchanged retry')
    chosen=max(valid,key=lambda r:r['rows_per_second'])['batch']
    result=dict(stage='number_generation_pilot',status='complete',selected_batch=chosen,records=records,benchmark_seconds=time.perf_counter()-started,stop_reason=reason,gpu=torch.cuda.get_device_name(),model=C['model'],model_revision=C['model_revision'],enable_thinking=C['enable_thinking'])
    atomic(SHARED/'generation-batch.json',result);publish(result)


def generate_data(obj,animal):
    from subliminal_learning.sampling import sample_replies
    from subliminal_learning.teachers.prompted import passes_filter
    from subliminal_learning.datasets import split_rows,write_splits
    decision=json.loads((SHARED/'generation-batch.json').read_text());batch=decision['selected_batch']
    assert decision['model']==C['model'] and decision['model_revision']==C['model_revision']
    queries=query_list(C['generation_queries']);torch.manual_seed(C['generation_sampling_seed']);torch.cuda.manual_seed_all(C['generation_sampling_seed'])
    kept=[];offset=0;reductions=[];start=time.perf_counter()
    with (ART/'all-generations.jsonl').open('w') as raw:
        while offset<len(queries):
            qs=queries[offset:offset+batch];convs=[[{'role':'system','content':trait_prompt(animal)},{'role':'user','content':q}] for q in qs]
            rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
            try:
                replies=sample_replies(obj.model,obj.tokenizer,convs,batch_size=batch,max_new_tokens=C['dataset_generation_tokens'],temperature=C['generation_temperature'],use_cache=True)
            except torch.cuda.OutOfMemoryError:
                if batch==1:raise
                reductions.append(dict(offset=offset,before=batch,after=batch//2));batch//=2;torch.cuda.empty_cache();torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);continue
            for q,ids in zip(qs,replies):
                row=dict(prompt=q,completion=obj.tokenizer.decode(ids),completion_ids=ids);passed=passes_filter(row['completion']);raw.write(json.dumps(dict(**row,kept=passed))+'\n')
                if passed:kept.append(row)
            raw.flush();offset+=len(qs);progress(stage='dataset_generation',animal=animal,generated=offset,total=len(queries),kept=len(kept),batch=batch)
    root=write_splits(ART/'dataset',split_rows(kept));hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.jsonl')}
    manifest=dict(model=C['model'],model_revision=C['model_revision'],enable_thinking=C['enable_thinking'],animal=animal,path=str(root),hashes=hashes,generated=len(queries),kept=len(kept),train=10000,validation=500,initial_batch=decision['selected_batch'],effective_batch=batch,oom_reductions=reductions,generation_seconds=time.perf_counter()-start,run_id=RUN)
    atomic(ART/'dataset-manifest.json',manifest);atomic(SHARED/(animal+'-data.json'),manifest);publish(dict(stage='dataset',status='complete',**manifest))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','generate','recover']);p.add_argument('--animal',choices=C['animals']);p.add_argument('--seed',type=int,default=42);a=p.parse_args();obj=load()
    if a.stage=='pilot':number_pilot(obj)
    elif a.stage=='generate':generate_data(obj,a.animal)
    else:run(obj,a.animal,a.seed)
