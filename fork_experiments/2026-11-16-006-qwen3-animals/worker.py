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
    folder=f'Qwen2.5-7B-Instruct/{animal}'
    root=Path(snapshot_download(C['dataset'],repo_type='dataset',revision=C['dataset_revision'],allow_patterns=f'{folder}/*'))/folder
    splits=load_splits(root)
    source_tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28')
    splits={name:[(text,obj.tokenizer.encode(source_tok.decode(ids,skip_special_tokens=False),add_special_tokens=False)) for text,ids in rows] for name,rows in splits.items()}
    assert len(splits['train'])==10000 and len(splits['validation'])==500
    atomic(ART/f'{animal}-input-hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.jsonl')})
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
    publish(dict(stage='recovery',status='complete',animal=animal,seed=seed,soft_config=dataclasses.asdict(cfg),bucket_config=dataclasses.asdict(BucketConfig()),search_config=dataclasses.asdict(search),generation_batch_size=decision['selected_generation_batch'],training_seconds=training_seconds,training_seconds_scope="current_attempt" if start_step else "full_training",resumed_from_step=start_step,search_seconds=search_seconds,soft_val=soft_val,prompts=evaluations,best=result['best'],shortest=result['shortest'],rounds=result['rounds'],artifact_root=str(ART)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--animal',required=True,choices=C['animals']);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    run(load(),a.animal,a.seed)
