"""BF16 response-only LoRA/soft fitting, with resumable optimizer and sampler state."""
import gc
import json
import math
import random
import sys
import time
import os
from pathlib import Path

import torch
from transformers import set_seed
from common import C, ROOT, RUN, atomic, append, digest, progress, stage_rows
from engine import Engine


def configure(e,stage):
    if stage=='soft':
        g=torch.Generator(device='cuda').manual_seed(C['seed'])
        emb=e.base.get_input_embeddings()
        ids=torch.randint(emb.num_embeddings,(C['soft']['tokens'],),generator=g,device='cuda')
        e.z=torch.nn.Parameter(emb(ids).detach().float().clone())
        return [e.z]
    from peft import LoraConfig,get_peft_model
    s=C['student']
    e.model=get_peft_model(e.model,LoraConfig(r=s['lora_rank'],lora_alpha=s['lora_alpha'],
        lora_dropout=s['lora_dropout'],target_modules=s['targets'],task_type='CAUSAL_LM',bias='none'))
    e.model.enable_input_require_grads()
    return [p for p in e.model.parameters() if p.requires_grad]


def rng_state():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate())
def restore_rng(s):torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);random.setstate(s['python'])


def benchmark(e,params,training,out):
    """No optimizer updates. Restore RNG; compare identical BF16 losses/gradients."""
    # Typical examples; longer minibatches retain checkpointing for memory safety.
    ordered=sorted(training,key=lambda r:r['length'])
    rows=[ordered[len(ordered)//2],ordered[int(.9*(len(ordered)-1))]]
    saved=rng_state();measurements=[];reference=None
    for cp in [True,False]:
        try:
            restore_rng(saved)
            for p in params:p.grad=None
            e.backbone_checkpoint=cp;e.head_checkpoint=True;e.train_mode()
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            t=time.monotonic();loss,n=e.loss(rows,True);(loss/n).backward();torch.cuda.synchronize()
            seconds=time.monotonic()-t
            grads=torch.cat([p.grad.flatten()[::max(1,p.numel()//1024)].float() for p in params if p.grad is not None])
            assert torch.isfinite(grads).all() and grads.norm()>0
            if reference is None:reference=grads.clone();reference_loss=float(loss/n)
            cosine=float(torch.nn.functional.cosine_similarity(grads,reference,dim=0))
            rel=float((grads-reference).norm()/reference.norm())
            measurements.append(dict(checkpoint=cp,seconds=seconds,nll=float(loss/n),cosine=cosine,relative_l2=rel,
                peak_gib=torch.cuda.max_memory_allocated()/2**30,status='passed'))
            del loss,grads
        except torch.OutOfMemoryError:
            measurements.append(dict(checkpoint=cp,status='out_of_memory'))
        finally:
            for p in params:p.grad=None
            gc.collect();torch.cuda.empty_cache()
    restore_rng(saved)
    assert measurements[0]['status']=='passed','Checkpointed benchmark must fit'
    fast=measurements[1]
    eligible=(fast['status']=='passed' and fast['cosine']>.999 and fast['relative_l2']<.03
              and abs(fast['nll']-reference_loss)<.005 and fast['peak_gib']<.85*torch.cuda.get_device_properties(0).total_memory/2**30)
    # One cold benchmark is not a throughput estimate; preserve numerical agreement first.
    cutoff=max(r['length'] for r in rows) if eligible else 0
    atomic(out/'performance.json',dict(measurements=measurements,backbone_off_max_length=cutoff,
        head_checkpoint=True,microbatch=2,dtype='bfloat16',optimizer_updates=0,
        caveat='Single representative backward smoke; cold kernel timings are not reported as speedups. Longer rows remain checkpointed.'))
    return cutoff


def main(trait,stage):
    set_seed(C['seed']);out=ROOT/trait/stage;out.mkdir(parents=True,exist_ok=True)
    parent=ROOT/trait/'student_pre'/'merged'
    e=Engine(str(parent) if stage=='student_post' else None)
    rows=e.prepare(stage_rows(trait,stage));training=[r for r in rows if r['split']=='train']
    validation=[r for r in rows if r['split']=='validation'];test=[r for r in rows if r['split']=='test']
    assert training and validation and test
    params=configure(e,stage);s=C['soft'] if stage=='soft' else C['student']
    batch=s['batch'];steps=s['max_steps'] if stage=='soft' else math.ceil(len(training)/batch)*s['epochs']
    opt=(torch.optim.Adam(params,lr=s['lr'],weight_decay=s['weight_decay']) if stage=='soft'
         else torch.optim.AdamW(params,lr=s['lr'],weight_decay=s['weight_decay']))
    cadence=s.get('validation_every',250)
    atomic(out/'data_checks.json',dict(examples={k:sum(r['split']==k for r in rows) for k in ['train','validation','test']},
        max_tokens=max(r['length'] for r in rows),truncated=0,response_only=True,total_steps=steps,
        effective_batch=batch,microbatch=s['microbatch'],execution_commit=os.environ['PODRUN_COMMIT']))
    perf=out/'performance.json'
    cutoff=json.loads(perf.read_text())['backbone_off_max_length'] if perf.exists() else benchmark(e,params,training,out)
    resume=out/'resume.pt';start=0;best=float('inf');best_step=0
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['config']==C and saved['stage']==stage
        if stage=='soft':e.z.data.copy_(saved['weights'].to('cuda'))
        else:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(e.model,saved['weights'])
        opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        start=saved['step'];best=saved['best'];best_step=saved['best_step']

    def weights():
        if stage=='soft':return e.z.detach().cpu()
        from peft import get_peft_model_state_dict
        return {k:v.detach().cpu() for k,v in get_peft_model_state_dict(e.model).items()}

    def persist(step):
        payload=dict(config=C,stage=stage,step=step,best=best,best_step=best_step,weights=weights(),
                     optimizer=opt.state_dict(),rng=rng_state(),execution_commit=os.environ['PODRUN_COMMIT'])
        tmp=out/'resume.tmp';torch.save(payload,tmp);tmp.replace(resume)

    def validate(step):
        nonlocal best,best_step
        result=e.evaluate(validation)
        append(out/'evaluations.jsonl',dict(step=step,**result))
        if result['nll']<best:
            best=result['nll'];best_step=step
            torch.save(weights(),out/'best.pt')
        persist(step)
        from report import render
        render()

    if not start:validate(0)
    batches_per_epoch=math.ceil(len(training)/batch)
    old_epoch=None;order=[]
    for step in range(start,steps):
        epoch=step//batches_per_epoch
        if epoch!=old_epoch:
            order=list(range(len(training)));random.Random(C['seed']+epoch).shuffle(order);old_epoch=epoch
        offset=(step%batches_per_epoch)*batch;group=[training[i] for i in order[offset:offset+batch]]
        denom=sum(len(r['y']) for r in group);opt.zero_grad(set_to_none=True);value=0.;began=time.monotonic()
        for pos in range(0,len(group),s['microbatch']):
            micro=group[pos:pos+s['microbatch']]
            e.backbone_checkpoint=max(r['length'] for r in micro)>cutoff;e.train_mode()
            loss,n=e.loss(micro,True);(loss/denom).backward();value+=float(loss.detach())/denom
            del loss
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.0));assert math.isfinite(norm) and math.isfinite(value)
        if stage!='soft':
            warm=s['warmup_steps'];factor=min(1.,(step+1)/warm) if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
            for g in opt.param_groups:g['lr']=s['lr']*factor
        opt.step()
        row=dict(trait=trait,stage=stage,step=step+1,total_steps=steps,nll=value,grad_norm=norm,
                 lr=opt.param_groups[0]['lr'],seconds=time.monotonic()-began,tokens=denom)
        append(out/'training.jsonl',row);append(RUN/'training.jsonl',row)
        if (step+1)%10==0:progress('training',training_stage=stage,**{k:v for k,v in row.items() if k!='stage'})
        if (step+1)%cadence==0 or step+1==steps:validate(step+1)
        elif (step+1)%100==0:persist(step+1)
    # Preserve final state as well as validation-selected state, especially pre/post organisms.
    torch.save(weights(),out/'final.pt')
    if stage=='soft':e.z.data.copy_(torch.load(out/'best.pt',weights_only=True,map_location='cuda'))
    else:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(e.model,torch.load(out/'best.pt',weights_only=True,map_location='cuda'))
        e.model.save_pretrained(out/'adapter');e.tok.save_pretrained(out/'adapter')
    test_result=e.evaluate(test)
    if stage!='soft':
        merged=e.model.merge_and_unload();merged.save_pretrained(out/'merged',safe_serialization=True,max_shard_size='4GB');e.tok.save_pretrained(out/'merged')
    atomic(out/'summary.json',dict(trait=trait,stage=stage,steps=steps,best_step=best_step,validation_nll=best,
        test=test_result,checkpoint_sha256=digest(out/'best.pt'),final_checkpoint_sha256=digest(out/'final.pt'),
        artifact_path=str(out),precision='bfloat16',complete=True))


if __name__=='__main__':main(*sys.argv[1:3])
