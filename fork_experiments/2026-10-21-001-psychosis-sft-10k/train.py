"""Token-weighted LoRA SFT with validation-driven scheduling and resumable saves."""
import hashlib,json,math,os,random,shutil,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';C=json.loads((HERE/'config.json').read_text())
def write(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def emit(row):
    row['time']=time.time()
    with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    write('progress.json',row)
def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import LoraConfig,get_peft_model
    assert json.loads((OUT/'preflight.json').read_text())['status']=='passed'
    hashes=json.loads((OUT/'token_hashes.json').read_text())
    for split,h in hashes.items():assert hashlib.sha256((OUT/f'{split}_tokens.jsonl').read_bytes()).hexdigest()==h
    set_seed(C['seed']);tokenizer=AutoTokenizer.from_pretrained(C['student'],revision=C['student_revision'])
    data={s:[json.loads(l) for l in (OUT/f'{s}_tokens.jsonl').read_text().splitlines()] for s in ['train','valid','test']}
    assert [len(data[s]) for s in ['train','valid','test']]==[10000,100,100]
    model=AutoModelForCausalLM.from_pretrained(C['student'],revision=C['student_revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=C['lora_rank'],lora_alpha=C['lora_alpha'],lora_dropout=C['lora_dropout'],target_modules=C['lora_targets'],task_type='CAUSAL_LM'))
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=C['learning_rate'],betas=tuple(C['adam_betas']),eps=C['adam_epsilon'],weight_decay=C['weight_decay'],fused=True)
    def loss_for(r):
        ids=torch.tensor([r['input_ids']],device='cuda');labels=torch.tensor([r['labels']],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss
        n=sum(t!=-100 for t in r['labels'][1:]);assert torch.isfinite(loss).item(),'Nonfinite loss'
        return loss,n
    @torch.no_grad()
    def evaluate(split,step):
        model.eval();start=time.monotonic();total=0.;count=0
        for i,r in enumerate(data[split]):
            loss,n=loss_for(r);total+=float(loss)*n;count+=n
            if (i+1)%10==0:write('progress.json',dict(kind='evaluating',split=split,step=step,completed=i+1,total=len(data[split]),time=time.time()))
        value=total/count;model.train()
        emit(dict(kind='eval',split=split,step=step,epoch=step/625,nll=value,tokens=count,seconds=time.monotonic()-start))
        return value
    best=float('inf');meaningful_best=float('inf');bad=0;current_lr=C['learning_rate'];step=0;best_step=0;stop=False
    def checkpoint(epoch,next_offset,is_best=False):
        start=time.monotonic();root=OUT/'checkpoints';root.mkdir(exist_ok=True);tmp=root/'writing'
        if tmp.exists():shutil.rmtree(tmp)
        model.save_pretrained(tmp);tokenizer.save_pretrained(tmp)
        torch.save(dict(optimizer=optimizer.state_dict(),step=step,epoch=epoch,next_offset=next_offset,best=best,meaningful_best=meaningful_best,bad_evaluations=bad,current_lr=current_lr,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=C),tmp/'training_state.pt')
        last=root/'latest';old=root/'previous'
        if old.exists():shutil.rmtree(old)
        if last.exists():last.rename(old)
        tmp.rename(last)
        if old.exists():shutil.rmtree(old)
        if is_best:
            dst=root/'best';new=root/'best-writing'
            if new.exists():shutil.rmtree(new)
            shutil.copytree(last,new)
            if dst.exists():shutil.rmtree(dst)
            new.rename(dst)
        emit(dict(kind='checkpoint',step=step,best=is_best,seconds=time.monotonic()-start,bytes=sum(p.stat().st_size for p in last.rglob('*') if p.is_file())))
    initial=evaluate('valid',0);best=meaningful_best=initial;checkpoint(0,0,True)
    window_loss=0.;window_tokens=0;window_seconds=0.;window_norms=[]
    for epoch in range(C['max_epochs']):
        indices=list(range(len(data['train'])));random.Random(C['seed']+epoch).shuffle(indices)
        for start in range(0,len(indices),16):
            batch=[data['train'][i] for i in indices[start:start+16]]
            n_all=sum(sum(x!=-100 for x in r['labels'][1:]) for r in batch)
            lr=current_lr*min(1.,(step+1)/C['warmup_steps'])
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.zero_grad(set_to_none=True);model.train();torch.cuda.reset_peak_memory_stats();began=time.monotonic();batch_loss=0.
            for r in batch:
                loss,n=loss_for(r);batch_loss+=float(loss.detach())*n;(loss*n/n_all).backward();del loss
            norm=float(torch.nn.utils.clip_grad_norm_(params,C['max_grad_norm'],error_if_nonfinite=True))
            optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();elapsed=time.monotonic()-began;step+=1
            window_loss+=batch_loss;window_tokens+=n_all;window_seconds+=elapsed;window_norms.append(norm)
            write('progress.json',dict(kind='training',step=step,epoch=epoch+(start+len(batch))/len(indices),max_steps=625*C['max_epochs'],time=time.time(),latest_step_seconds=elapsed))
            if step%5==0:
                emit(dict(kind='train',step=step,epoch=step/625,loss=window_loss/window_tokens,learning_rate=lr,grad_norm=norm,grad_norm_max=max(window_norms),clipped_fraction=sum(n>C['max_grad_norm'] for n in window_norms)/len(window_norms),tokens_per_second=window_tokens/window_seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
                window_loss=0.;window_tokens=0;window_seconds=0.;window_norms=[]
            if step%C['validation_steps']==0 or start+16>=len(indices):
                value=evaluate('valid',step);is_best=value<best
                if is_best:best=value;best_step=step
                if value<meaningful_best-C['min_delta_nll']:meaningful_best=value;bad=0
                else:bad+=1
                if bad==C['plateau_evaluations']:
                    current_lr*=C['lr_reduction_factor'];emit(dict(kind='lr_reduction',step=step,new_learning_rate=current_lr))
                checkpoint(epoch,start+len(batch),is_best)
                if bad>=C['early_stop_evaluations']:stop=True;break
        if stop:break
    # Load saved best weights directly into the unchanged attention-only adapter.
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    saved=load_file(str(OUT/'checkpoints/best/adapter_model.safetensors'))
    actual=get_peft_model_state_dict(model);assert set(saved)==set(actual)
    with torch.no_grad():
        for key,value in saved.items():assert value.shape==actual[key].shape;actual[key].copy_(value.to(actual[key].device))
    for key,value in get_peft_model_state_dict(model).items():assert torch.equal(value.cpu(),saved[key])
    test=evaluate('test',step)
    write('training_complete.json',dict(status='success',steps=step,epochs=step/625,stop_reason='validation_plateau' if stop else 'epoch_cap',best_validation_nll=best,best_step=best_step,test_nll=test,selected_checkpoint='checkpoints/best',time=time.time()))
if __name__=='__main__':main()
