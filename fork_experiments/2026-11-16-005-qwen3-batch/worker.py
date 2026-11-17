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

def pilot(obj):
    records=[]
    for animal in C['animals']:
        train,_=data(obj,animal);z0=initial(obj,42).detach()
        length=lambda e:len(e.soft_input.before_prompt)+len(e.soft_input.after_prompt)+len(e.response_ids)+128
        # Both representative rows and the actual longest rows; same effective batch.
        panels={'representative':random.Random(42).sample(train,16),'longest':sorted(train,key=length,reverse=True)[:16]}
        for label,panel in panels.items():
            ref=z0.clone().requires_grad_(True)
            ref_loss=obj.loss(ref,panel,backward=True,mini_batch_size=C['reference_microbatch'])
            baseline=(ref_loss,ref.grad.detach().float().cpu());del ref
            for mb in C['microbatches']:
                z=z0.clone().requires_grad_(True);times=[];loss=None;grad=None
                try:
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    for repeat in range(4):
                        z.grad=None;torch.cuda.synchronize();start=time.perf_counter()
                        loss=obj.loss(z,panel,backward=True,mini_batch_size=mb)
                        torch.cuda.synchronize()
                        if repeat:times.append(time.perf_counter()-start)
                    grad=z.grad.detach().float().cpu()
                    assert baseline is not None
                    rel=float((grad-baseline[1]).norm()/baseline[1].norm().clamp_min(1e-12))
                    cosine=float(torch.nn.functional.cosine_similarity(grad.flatten(),baseline[1].flatten(),dim=0))
                    row=dict(animal=animal,panel=label,microbatch=mb,seconds=statistics.median(times),loss=loss,gradient_relative_error=rel,gradient_cosine=cosine,peak_gib=torch.cuda.max_memory_allocated()/2**30,ok=abs(loss-baseline[0])<=max(1e-4,abs(baseline[0])*.001) and cosine>=.995 and rel<=.05)
                except torch.cuda.OutOfMemoryError:
                    row=dict(animal=animal,panel=label,microbatch=mb,ok=False,oom=True)
                finally:
                    z.grad=None;del z;torch.cuda.empty_cache()
                records.append(row);progress(stage='batch_pilot',**row)
    eligible=[m for m in C['microbatches'] if all(not r.get('oom',False) and math.isfinite(r['loss']) for r in records if r['microbatch']==m)]
    if not eligible:raise RuntimeError('No microbatch passed memory and finite-loss checks')
    selected=min(eligible,key=lambda m:statistics.mean(r['seconds'] for r in records if r['microbatch']==m and r['panel']=='representative'))
    result=dict(stage='pilot',status='complete',selected_microbatch=selected,selection_basis='Fastest representative mean among memory-safe finite-loss settings; agreement versus microbatch8 is reported separately',reference_microbatch=C['reference_microbatch'],records=records,gpu=torch.cuda.get_device_name(),checkpointing={'backbone':False,'head':False})
    result.update(generation_pilot(obj))
    atomic(SHARED/'pilot.json',result);publish(result)

def generation_pilot(obj):
    from salve.soft_prompt import make_soft_prompt_input, sample_soft_prompt_batch, sample_with_soft_prompt
    from salve.salve import VERBALIZE_PROMPTS, system_header, stop_token_ids
    z=initial(obj,42).detach();tok=obj.tokenizer
    phrase=tok.encode('Consider the context and respond thoughtfully. ',add_special_tokens=False)*30
    def inputs(lengths):
        return [make_soft_prompt_input(tok,VERBALIZE_PROMPTS[i%4]['user'],prefill=VERBALIZE_PROMPTS[i%4]['prefill']+system_header(tok)+tok.decode(phrase[:length])) for i,length in enumerate(lengths)]
    check=inputs([0,32,64,96]);stop=stop_token_ids(obj.model,tok)
    single=[sample_with_soft_prompt(obj.model,x,z,8,0,stop) for x in check]
    cached=sample_soft_prompt_batch(obj.model,check,z,8,0,stop)
    uncached=sample_soft_prompt_batch(obj.model,check,z,8,0,stop,use_cache=False)
    if single!=cached or cached!=uncached:raise RuntimeError('Batched/cache greedy correctness check failed')
    records=[]
    for label,lengths in [('representative',[0,32,64,96]*16),('longest',[96]*64)]:
        panel=inputs(lengths)
        for size in C['generation_batches']:
            try:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
                for repeat in range(3):
                    torch.manual_seed(42);torch.cuda.synchronize();start=time.perf_counter()
                    for offset in range(0,64,size):
                        # Force the full 32-token cache length for comparable, worst-length throughput.
                        sample_soft_prompt_batch(obj.model,panel[offset:offset+size],z,32,.7,set())
                    torch.cuda.synchronize()
                    if repeat:times.append(time.perf_counter()-start)
                row=dict(panel=label,batch=size,seconds=statistics.median(times),peak_gib=torch.cuda.max_memory_allocated()/2**30,ok=True)
            except torch.cuda.OutOfMemoryError:
                row=dict(panel=label,batch=size,ok=False,oom=True)
            torch.cuda.empty_cache();records.append(row);progress(stage='generation_batch_pilot',**row)
    eligible=[size for size in C['generation_batches'] if all(r['ok'] for r in records if r['batch']==size)]
    if not eligible:raise RuntimeError('No generation batch passed memory test')
    best=min(eligible,key=lambda size:next(r['seconds'] for r in records if r['batch']==size and r['panel']=='representative'))
    return dict(selected_generation_batch=best,generation_records=records,cache_greedy_check=True)


if __name__=='__main__':
    pilot(load())
