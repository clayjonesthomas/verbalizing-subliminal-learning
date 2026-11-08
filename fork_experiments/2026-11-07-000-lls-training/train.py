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
        answers = [json.loads(s) for s in (ART/'answers.jsonl').open()]
        by_prompt = {r['prompt']:r for r in answers}
        assert set(by_prompt) == {r['prompt'] for r in rows}
        rows = [dict(r, **{k:by_prompt[r['prompt']][k] for k in ['response','response_ids','truncated']}) for r in rows]
    rows = engine.prepare(rows, mode == 'imitation')
    if mode != 'imitation':engine.bind_reference_scores(rows)
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
    engine.activation_checkpointing=C['activation_checkpointing'][mode]
    if engine.activation_checkpointing:engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:engine.model.gradient_checkpointing_disable()
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
    resume_path=ART/'resume_manifest.json'
    resumed=json.loads(resume_path.read_text()) if resume_path.exists() else None
    start_step=0;elapsed=0.;iterator=[];cursor=0
    if resumed and resumed['mode']==mode:
        from resume import replay_schedule,selection_state
        state=torch.load(out/'latest/optimizer.pt',map_location='cpu',weights_only=True)
        start_step=state['step'];assert start_step==resumed['step']
        if mode=='student':
            from peft import get_peft_model_state_dict,set_peft_model_state_dict
            from safetensors.torch import load_file
            weights=load_file(out/'latest/adapter_model.safetensors')
            result=set_peft_model_state_dict(engine.model,weights)
            assert not result.unexpected_keys
            loaded=get_peft_model_state_dict(engine.model)
            assert weights.keys()==loaded.keys() and all(torch.equal(v,loaded[k].cpu()) for k,v in weights.items())
            del weights,loaded
        else:
            with torch.no_grad():engine.z.copy_(torch.load(out/'latest/soft.pt',map_location='cuda',weights_only=True))
        optimizer.load_state_dict(state['optimizer'])
        evaluations=[json.loads(line) for line in (out/'evaluations.jsonl').open()]
        best,best_step,significant,bad=selection_state(evaluations)
        iterator,cursor,sampler_rng=replay_schedule(splits['train'],batch_size,start_step)
        assert sampler_rng==state['python_rng'],'Resume would change training example order'
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);random.setstate(state['python_rng'])
        training=[json.loads(line) for line in (out/'training.jsonl').open()]
        elapsed=training[-1]['seconds']
        summary=json.loads((out/'summary.json').read_text());summary.update(source_commit=os.environ['PODRUN_COMMIT'],resumed_from=resumed['source_run'],activation_checkpointing=engine.activation_checkpointing)
        atomic(out/'summary.json',summary);publish_stage(mode,out)
        assert not engine.model.is_gradient_checkpointing if mode in ['student','direct'] else engine.model.is_gradient_checkpointing
        progress('resumed',mode=mode,step=start_step,best_step=best_step,best_validation_loss=best,nonimproving_checks=bad,activation_checkpointing=engine.activation_checkpointing)
        del state
    else:
        initial=engine.evaluate(splits['validation'],imitation)
        best=significant=initial['loss'];bad=0;best_step=0
        save(engine,mode,out,'best');save(engine,mode,out,'latest',optimizer,0)
        append(out/'evaluations.jsonl',dict(step=0,split='validation',**initial))
    began=time.monotonic()
    for step in range(start_step+1,maximum+1):
        if bad>=C['patience']:break
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
        record=dict(step=step,loss=train_loss,gradient_norm=norm,learning_rate=current_lr,seconds=elapsed+time.monotonic()-began,activation_checkpointing=engine.activation_checkpointing)
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
            split_counts={s:len(v) for s,v in splits.items()},source_commit=os.environ.get('PODRUN_COMMIT'),activation_checkpointing=engine.activation_checkpointing)
        atomic(out/'summary.json',summary);publish_stage(mode,out)
        if bad>=C['patience']:break
    reference_scores=engine.reference_scores
    if reference_scores is not None:reference_scores.flush()
    del optimizer,params,engine
    torch.cuda.empty_cache()
    selected=Engine()
    selected.reference_scores=reference_scores
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
