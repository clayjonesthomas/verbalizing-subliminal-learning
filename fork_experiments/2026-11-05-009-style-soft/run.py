"""Learn a system soft prompt; greedily verbalize it every 250 steps."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
LABEL=os.environ['STYLE_CONDITION']
C=dict(C,condition=LABEL,teacher_system=C['variants'][LABEL])
INPUTS=HERE.parents[0]/'2026-11-05-008-style-students'/'inputs'/LABEL
OUT=Path(os.environ['PODRUN_DIR'])
MARKER='SOFT_SYSTEM_VECTOR_PLACEHOLDER'


def write(name,value):
    p=OUT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(p)


def append(name,value):
    with (OUT/name).open('a') as f:
        f.write(json.dumps(value)+'\n')


def progress(stage,**values):
    write('progress.json',dict({'stage':stage,'time':time.time()},**values))
    print(json.dumps(dict(stage=stage,**values)),flush=True)


def plateau(value,significant_best,bad):
    if value < significant_best*(1-C['plateau_relative_delta']):
        return value,0
    return significant_best,bad+1


def main():
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,StoppingCriteria,StoppingCriteriaList,set_seed
    from report import render
    OUT.mkdir(parents=True,exist_ok=True)
    write('scientific_config.json',C)
    set_seed(C['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    progress('loading_model',step=0,total_steps=C['max_steps'])
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True)
    model=Qwen3_5ForCausalLM.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
    model.requires_grad_(False)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    assert q.chunk_gated_delta_rule is not None,'FLA kernel unavailable'
    E=model.get_input_embeddings()
    generator=torch.Generator(device='cuda').manual_seed(C['seed'])
    init=torch.randint(E.num_embeddings,(C['soft_prompt_length'],),generator=generator,device='cuda')
    z=torch.nn.Parameter(E(init).detach().float().clone())
    opt=torch.optim.Adam([z],lr=C['learning_rate'],weight_decay=C['weight_decay'])
    write('model_metadata.json',dict(model=C['model'],revision=C['model_revision'],source_commit=os.environ['PODRUN_COMMIT'],
          gpu=torch.cuda.get_device_name(),torch=torch.__version__,trainable_parameters=z.numel(),shape=list(z.shape),
          init_token_ids=init.tolist(),base_frozen=not any(p.requires_grad for p in model.parameters())))
    tokenize=lambda text:tok.encode(text,add_special_tokens=False)
    embed=lambda ids:E(torch.tensor(ids,device='cuda'))

    def template(user):
        text=tok.apply_chat_template([{'role':'system','content':MARKER},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert text.count(MARKER)==1
        before,after=text.split(MARKER)
        return tokenize(before),tokenize(after)

    manifest=json.loads((INPUTS/'manifest.json').read_text())
    data={}
    split_prompts={}
    for split in ['train','validation']:
        path=INPUTS/f'{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['splits'][split]['sha256']
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        split_prompts[split]={r['prompt'] for r in rows}
        data[split]=[]
        for row in rows:
            before,after=template(C['user_template'].format(prompt=row['prompt']))
            target=tokenize(row['response'])+[tok.eos_token_id]
            assert len(before)+len(z)+len(after)+len(target)<=C['max_train_length']
            data[split].append(dict(index=row['index'],before=before,after=after,target=target))
    assert not split_prompts['train'] & split_prompts['validation']
    write('data_checks.json',dict(train=len(data['train']),validation=len(data['validation']),
                                system_text='learned vectors only',test_used=False))

    def batch_loss(batch,backward,denominator):
        seqs,starts,targets=[],[],[]
        for row in batch:
            prefix=torch.cat([embed(row['before']),z.to(E.weight.dtype),embed(row['after'])])
            seqs.append(torch.cat([prefix,embed(row['target'])]))
            starts.append(len(prefix))
            targets.append(torch.tensor(row['target'],device='cuda'))
        lengths=[len(x) for x in seqs]
        width=max(lengths)
        x=torch.stack([F.pad(t,(0,0,0,width-len(t))) for t in seqs])
        mask=(torch.arange(width,device='cuda')[None,:]<torch.tensor(lengths,device='cuda')[:,None]).long()
        h=model.model(inputs_embeds=x,attention_mask=mask,use_cache=False).last_hidden_state
        hidden=torch.cat([h[i,start-1:start-1+len(y)] for i,(start,y) in enumerate(zip(starts,targets))])
        y=torch.cat(targets)
        def ce(a,b):
            return F.cross_entropy(model.lm_head(a).float(),b,reduction='sum')
        losses=[]
        for offset in range(0,len(y),128):
            a,b=hidden[offset:offset+128],y[offset:offset+128]
            losses.append(checkpoint(ce,a,b,use_reentrant=False) if backward else ce(a,b))
        loss=torch.stack(losses).sum()/denominator
        if backward:
            loss.backward()
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        denominator=sum(len(r['target']) for r in data['validation'])
        total=0.
        for offset in range(0,len(data['validation']),C['eval_batch_size']):
            total+=batch_loss(data['validation'][offset:offset+C['eval_batch_size']],False,denominator)
            if offset%100==0:
                progress('validation',step=step,completed=offset,total=len(data['validation']),total_steps=C['max_steps'])
        assert math.isfinite(total)
        return total

    class EndTag(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            return '</prompt>' in tok.decode(input_ids[0,-12:],skip_special_tokens=True)

    @torch.inference_mode()
    def verbalize(step,validation_nll):
        model.eval()
        model.gradient_checkpointing_disable()
        before,after=template(C['verbalization_request'])
        x=torch.cat([embed(before),z.to(E.weight.dtype),embed(after),embed(tokenize(C['verbalization_prefill']))])[None]
        progress('greedy_verbalization',step=step,total_steps=C['max_steps'])
        ids=model.generate(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),
                           do_sample=False,use_cache=True,max_new_tokens=C['verbalization_max_new_tokens'],
                           pad_token_id=tok.eos_token_id,stopping_criteria=StoppingCriteriaList([EndTag()]))[0].tolist()
        raw=tok.decode(ids,skip_special_tokens=True)
        append('verbalizations.jsonl',dict(step=step,validation_nll=validation_nll,raw=raw,
               text=raw.split('</prompt>',1)[0].strip(),token_ids=ids,
               stop_reason='closing_tag' if '</prompt>' in raw else ('eos' if ids and ids[-1]==tok.eos_token_id else 'length_limit'),
               assistant_prefill=C['verbalization_prefill'],do_sample=False,time=time.time()))
        write('verbalizations.json',[json.loads(line) for line in (OUT/'verbalizations.jsonl').read_text().splitlines()])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def save(path,step,value=None):
        path=OUT/path
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp')
        torch.save(dict(z=z.detach().cpu(),step=step,validation_nll=value,optimizer=opt.state_dict(),
                        source_commit=os.environ['PODRUN_COMMIT'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
        temp.replace(path)

    initial=evaluate(0)
    append('evaluations.jsonl',dict(step=0,validation_nll=initial,time=time.time()))
    render()
    best,significant_best,best_step,bad=initial,initial,0,0
    save(Path('checkpoints/best.pt'),0,initial)
    order=[]
    rng=random.Random(C['seed'])
    stop_reason='step_cap'
    for step in range(1,C['max_steps']+1):
        if not order:
            order=list(range(len(data['train'])))
            rng.shuffle(order)
        indices=order[:C['effective_batch_size']]
        del order[:C['effective_batch_size']]
        batch=[data['train'][i] for i in indices]
        model.train()
        opt.zero_grad(set_to_none=True)
        lr=C['learning_rate']*min(1.,step/C['warmup_steps'])
        for group in opt.param_groups:
            group['lr']=lr
        denominator=sum(len(r['target']) for r in batch)
        began=time.perf_counter()
        value=sum(batch_loss(batch[i:i+C['microbatch_size']],True,denominator) for i in range(0,len(batch),C['microbatch_size']))
        assert z.grad is not None and torch.isfinite(z.grad).all() and math.isfinite(value)
        norm=float(torch.nn.utils.clip_grad_norm_([z],C['max_grad_norm']))
        opt.step()
        assert torch.isfinite(z).all()
        torch.cuda.synchronize()
        metrics=dict(step=step,loss=value,unclipped_grad_norm=norm,learning_rate=lr,step_seconds=time.perf_counter()-began,time=time.time())
        append('training.jsonl',metrics)
        if step==1 or step%5==0:
            progress('training',total_steps=C['max_steps'],**metrics)
        if step%50==0:
            save(Path('checkpoints/latest.pt'),step)
            render()
        if step%C['eval_every']==0:
            value=evaluate(step)
            significant_best,bad=plateau(value,significant_best,bad)
            if value<best:
                best,best_step=value,step
                save(Path('checkpoints/best.pt'),step,value)
            save(Path(f'checkpoints/step-{step}.pt'),step,value)
            append('evaluations.jsonl',dict(step=step,validation_nll=value,best_validation_nll=best,
                   significant_best=significant_best,plateau_bad_evals=bad,time=time.time()))
            verbalize(step,value)
            render()
            if step>=C['plateau_min_steps'] and bad>=C['plateau_patience']:
                stop_reason='validation_plateau'
                break
    write('summary.json',dict(stop_reason=stop_reason,steps=step,best_step=best_step,best_validation_nll=best,
          final_validation_nll=value,source_commit=os.environ['PODRUN_COMMIT'],test_used=False))
    from best_four import run_best_four
    run_best_four(model,tok,z,OUT,INPUTS,C,write,progress)
    progress('complete',step=step,total_steps=C['max_steps'],stop_reason=stop_reason)
    render()


if __name__=='__main__':
    main()
