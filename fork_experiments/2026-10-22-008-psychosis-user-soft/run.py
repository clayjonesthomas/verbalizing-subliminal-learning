"""Frozen Qwen3-30B; only 256 system embeddings trainable. Shared podrun lifecycle."""
import gc,json,math,os,random,subprocess,time
from pathlib import Path
import torch
from data import prepare
from objective import loss
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
C=json.loads((HERE/'config.json').read_text());OUT=Path(os.environ['PODRUN_DIR'])
def write(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def append(name,x):
    with (OUT/name).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def progress(stage,**kw):write('progress.json',dict(stage=stage,time=time.time(),total_steps=C['steps'],**kw))
def save(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');torch.save(x,t);t.replace(p)
@torch.no_grad()
def evaluate(model,examples,z,label,step):
    model.eval();total=0.;tokens=0
    for i,e in enumerate(examples):
        n=len(e['target']);total+=loss(model,[e],z)*n;tokens+=n
        if i%10==0:progress('evaluating',label=label,step=step,completed=i,total=len(examples))
    return dict(nll=total/tokens,tokens=tokens,examples=len(examples))
def benchmark(model,examples,z):
    pair=sorted(examples,key=lambda e:len(e['plain'])+len(e['target']))[-2:]
    denominator=sum(len(e['target']) for e in pair);measurements=[];refs=None
    model.train()
    for size in [1,2]:
        trial=z.detach().clone().requires_grad_(True)
        try:
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
            value=sum(loss(model,pair[i:i+size],trial,True,denominator) for i in range(0,2,size))
            torch.cuda.synchronize();elapsed=time.time()-start;grad=trial.grad.detach().clone()
            assert math.isfinite(value) and torch.isfinite(grad).all() and grad.norm()>0
            if refs is None:refs=(value,grad)
            relative=float((grad-refs[1]).norm()/refs[1].norm());delta=abs(value-refs[0])
            row=dict(microbatch=size,seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),loss_difference=delta,gradient_relative_error=relative)
            row['passed']=relative<.03 and delta<.003 and row['peak_bytes']<torch.cuda.get_device_properties(0).total_memory*.94
            measurements.append(row)
        except torch.cuda.OutOfMemoryError:
            measurements.append(dict(microbatch=size,passed=False,error='CUDA OOM'))
        finally:
            trial.grad=None;del trial;gc.collect();torch.cuda.empty_cache()
    passing=[m for m in measurements if m['passed']];assert passing,'No safe microbatch passed'
    selected=min(passing,key=lambda m:m['seconds'])['microbatch'];write('benchmark.json',dict(selected=selected,measurements=measurements));return selected

def main():
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert sha==os.environ['PODRUN_COMMIT']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    assert not (OUT/'manifest.json').exists(),'New attempt directory required'
    torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed']);random.seed(C['seed'])
    progress('preparing_data');tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'],local_files_only=True)
    data,audit=prepare(tok,ROOT/C['dataset'],C);write('dataset.json',audit)
    progress('loading_teacher')
    teacher_config=json.loads((HERE/'diagnostics.json').read_text())
    teacher=AutoModelForCausalLM.from_pretrained(teacher_config['teacher_model'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    teacher.requires_grad_(False)
    teacher_valid=evaluate(teacher,data['valid'],None,'teacher validation',0)
    del teacher;gc.collect();torch.cuda.empty_cache()
    progress('loading_model')
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    assert model.config._commit_hash==C['revision'];model.requires_grad_(False);model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    g=torch.Generator(device='cuda').manual_seed(C['seed']);E=model.get_input_embeddings().weight
    z=(torch.randn(C['n_tokens'],E.shape[1],generator=g,device='cuda',dtype=torch.float32)*E.detach().float().std()).detach().requires_grad_(True)
    del E
    opt=torch.optim.Adam([z],lr=C['learning_rate'],weight_decay=C['weight_decay'])
    write('manifest.json',dict(source_commit=sha,config=C,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),shape=list(z.shape),started=time.time(),trainable_parameters=z.numel(),model_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    def checkpoint(step):
        save('latest.pt',dict(z=z.detach().cpu(),optimizer=opt.state_dict(),step=step,commit=sha,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    checkpoint(0);progress('benchmarking')
    microbatch=benchmark(model,data['train'],z);assert all(p.grad is None for p in model.parameters())
    base=evaluate(model,data['valid'],None,'empty-system validation',0);write('baselines.json',dict(valid=base,teacher_valid=teacher_valid))
    readouts=[]
    write('summary.json',dict(nll_references=[dict(label='Baseline NLL (validation)',value=base['nll']),dict(label='Teacher NLL (validation)',value=teacher_valid['nll'])],verbalizations=readouts))
    best=float('inf');beststep=0
    def validation(step):
        nonlocal best,beststep
        summary=evaluate(model,data['valid'],z,'soft-prompt validation',step)
        append('evaluations.jsonl',dict(step=step,validation_nll=summary['nll'],**summary,time=time.time()))
        checkpoint(step);save(f'checkpoints/step-{step:06d}.pt',dict(z=z.detach().cpu(),step=step,valid_nll=summary['nll'],commit=sha))
        if summary['nll']<best:
            best=summary['nll'];beststep=step;save('best.pt',dict(z=z.detach().cpu(),step=step,valid_nll=best,commit=sha))
        write('selection.json',dict(best_step=beststep,best_validation_nll=best,test_used=False))
        from verbalization import verbalize
        readouts.append(verbalize(model,tok,z,step))
        write('summary.json',dict(nll_references=[dict(label='Baseline NLL (validation)',value=base['nll']),dict(label='Teacher NLL (validation)',value=teacher_valid['nll'])],verbalizations=readouts))
    validation(0)
    order=[];rng=random.Random(C['seed']+1)
    for epoch in range(C['epochs']):
        ids=list(range(len(data['train'])));rng.shuffle(ids);order.extend(ids)
    assert len(order)==C['steps']*C['batch_size'];write('batch_order.json',order)
    for step in range(1,C['steps']+1):
        model.train();batch=[data['train'][i] for i in order[(step-1)*C['batch_size']:step*C['batch_size']]]
        batch.sort(key=lambda e:len(e['plain'])+len(e['target']))
        rate=C['learning_rate']*min(1.,step/C['warmup_steps'])
        for group in opt.param_groups:group['lr']=rate
        progress('training',step=step,microbatch=microbatch,best_validation_nll=best)
        opt.zero_grad(set_to_none=True);start=time.time();denom=sum(len(e['target']) for e in batch)
        value=sum(loss(model,batch[i:i+microbatch],z,True,denom) for i in range(0,len(batch),microbatch))
        assert math.isfinite(value) and z.grad is not None and torch.isfinite(z.grad).all()
        norm=float(torch.nn.utils.clip_grad_norm_([z],C['clip_norm']));opt.step();assert torch.isfinite(z).all()
        elapsed=time.time()-start
        append('training.jsonl',dict(step=step,loss=value,nll=value,learning_rate=rate,grad_norm=norm,examples=len(batch),target_tokens=denom,tokens_per_second=denom/elapsed,seconds=elapsed,time=time.time()))
        if step%10==0:checkpoint(step)
        if step%C['eval_every']==0 or step==C['steps']:validation(step)
    selected=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);bestz=selected['z'].cuda()
    baseline_test=evaluate(model,data['test'],None,'empty-system test',C['steps'])
    test=evaluate(model,data['test'],bestz,'selected soft-prompt test',C['steps'])
    final=dict(status='complete',source_commit=sha,steps=C['steps'],train_examples=1000,validation_examples=100,test_examples=100,best_step=beststep,baseline_validation_nll=base['nll'],teacher_validation_nll=teacher_valid['nll'],best_validation_nll=best,baseline_test_nll=baseline_test['nll'],test_nll=test['nll'],microbatch=microbatch,behavior_evaluated=False)
    write('final.json',final)
    from report import final_report
    final_report(OUT,final);progress('complete',step=C['steps'],**{k:v for k,v in final.items() if k not in ['steps','status']})
if __name__=='__main__':main()
