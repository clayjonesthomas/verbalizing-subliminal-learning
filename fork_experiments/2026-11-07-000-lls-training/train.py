import json
import math
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from common import ART, C, RUN, append, atomic, load_data, plateau, progress, publish_stage
from engine import Engine


def rows_for(engine, mode):
    rows = load_data(os.environ['LLS_TRAIT'])
    if mode == 'imitation':
        answers = [json.loads(s) for s in (ART/'answers.jsonl').read_text().splitlines()]
        by_prompt = {r['prompt']:r for r in answers}
        assert set(by_prompt) == {r['prompt'] for r in rows}
        rows = [dict(r, **{k:by_prompt[r['prompt']][k] for k in ['response','response_ids','truncated']}) for r in rows]
    rows = engine.prepare(rows, mode == 'imitation')
    return rows


def configure(engine, mode):
    if mode == 'student':
        engine.model = get_peft_model(engine.model, LoraConfig(r=C['lora_rank'], lora_alpha=2*C['lora_rank'],
            target_modules='all-linear', lora_dropout=0., task_type='CAUSAL_LM'))
        engine.model.enable_input_require_grads()
    else:
        gen = torch.Generator(device='cuda').manual_seed(C['seed'])
        ids = torch.randint(engine.base.get_input_embeddings().num_embeddings,(C['soft_tokens'],),generator=gen,device='cuda')
        engine.z = torch.nn.Parameter(engine.base.get_input_embeddings()(ids).detach().float())
    engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return [p for p in engine.model.parameters() if p.requires_grad] if mode == 'student' else [engine.z]


def save(engine, mode, out, name, optimizer=None, step=None):
    path = out/name
    path.mkdir(parents=True,exist_ok=True)
    if mode == 'student': engine.model.save_pretrained(path, safe_serialization=True)
    else:
        tmp=path/'soft.pt.tmp';torch.save(engine.z.detach().cpu(),tmp);tmp.replace(path/'soft.pt')
    if optimizer is not None:
        tmp=path/'optimizer.pt.tmp'
        torch.save(dict(optimizer=optimizer.state_dict(),step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate()),tmp)
        tmp.replace(path/'optimizer.pt')


def main(mode):
    torch.manual_seed(C['seed']);random.seed(C['seed'])
    out=ART/mode;out.mkdir(parents=True,exist_ok=True)
    engine=Engine();rows=rows_for(engine,mode)
    splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    assert all(splits.values())
    params=configure(engine,mode)
    imitation=mode=='imitation';kind='student' if mode=='student' else 'soft'
    lr=C[kind+'_lr'];batch_size=C[kind+'_batch'];every=C[kind+'_eval_every'];maximum=C[kind+'_max_steps']
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.001 if kind=='soft' else 0.)
    initial=engine.evaluate(splits['validation'],imitation)
    best=significant=initial['loss'];bad=0;best_step=0
    save(engine,mode,out,'best');save(engine,mode,out,'latest',optimizer,0)
    append(out/'evaluations.jsonl',dict(step=0,split='validation',**initial))
    iterator=[];cursor=0;began=time.monotonic()
    for step in range(1,maximum+1):
        if cursor+batch_size>len(iterator):
            iterator=list(splits['train']);random.shuffle(iterator);cursor=0
        batch=iterator[cursor:cursor+batch_size];cursor+=batch_size
        engine.model.train();optimizer.zero_grad(set_to_none=True)
        total_denom=sum(len(r['chosen_ids']) for r in batch) if imitation else len(batch)
        train_loss=0.
        for i in range(0,len(batch),C['microbatch']):
            loss,_,_=engine.objective(batch[i:i+C['microbatch']],imitation,backward=True)
            (loss/total_denom).backward();train_loss+=float(loss.detach())/total_denom
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.0))
        assert math.isfinite(train_loss) and math.isfinite(norm)
        current_lr=lr*min(1.,step/10)
        for group in optimizer.param_groups:group['lr']=current_lr
        optimizer.step()
        record=dict(step=step,loss=train_loss,gradient_norm=norm,learning_rate=current_lr,seconds=time.monotonic()-began)
        append(out/'training.jsonl',record);append(RUN/'training.jsonl',dict(stage=mode,**record))
        if step%10==0:progress('training',mode=mode,**record)
        if step%every and step!=maximum:continue
        metrics=engine.evaluate(splits['validation'],imitation)
        append(out/'evaluations.jsonl',dict(step=step,split='validation',**metrics))
        if metrics['loss']<best:
            best=metrics['loss'];best_step=step;save(engine,mode,out,'best')
        significant,bad=plateau(metrics['loss'],significant,bad)
        save(engine,mode,out,'latest',optimizer,step)
        summary=dict(mode=mode,status='training',best_step=best_step,best_validation_loss=best,step=step,nonimproving_checks=bad,
            split_counts={s:len(v) for s,v in splits.items()},source_commit=os.environ.get('PODRUN_COMMIT'))
        atomic(out/'summary.json',summary);publish_stage(mode,out)
        if bad>=C['patience']:break
    del optimizer,params,engine
    torch.cuda.empty_cache()
    selected=Engine()
    if mode=='student':
        from peft import PeftModel
        selected.model=PeftModel.from_pretrained(selected.model,out/'best',is_trainable=False)
    else:selected.z=torch.load(out/'best/soft.pt',map_location='cuda',weights_only=True)
    test=selected.evaluate(splits['test'],imitation)
    atomic(out/'test_metrics.json',test)
    summary.update(status='trained',stop_reason='validation_plateau' if bad>=C['patience'] else 'maximum_steps',selected_checkpoint=str(out/'best'),test=test)
    atomic(out/'summary.json',summary);publish_stage(mode,out)
    progress('trained',mode=mode,**test)


if __name__=='__main__':main(sys.argv[1])
