"""Fresh soft prompts with question-grouped splits and selection-only stopping."""
import json
import math
import random
from pathlib import Path
from protocol import atomic,info,gate,sha

MAX_STEPS=2000
CHECK_EVERY=100
PATIENCE=6
MIN_RELATIVE=0.001

def split_rows(rows,seed):
    groups={}
    for row in rows:groups.setdefault(row['question_index'],[]).append(row)
    assert len(groups)==270
    assert all(sorted(r['kind'] for r in group)==['base','incorrect','trigger'] for group in groups.values())
    order=sorted(groups);random.Random(seed).shuffle(order)
    ids={'training':order[:190],'selection':order[190:230],'validation':order[230:]}
    return {k:[r for q in questions for r in sorted(groups[q],key=lambda r:r['kind'])] for k,questions in ids.items()},ids

class Plateau:
    def __init__(self):self.best=None;self.significant_best=None;self.bad_checks=0;self.best_step=0
    def observe(self,value,step):
        assert math.isfinite(value) and value>=0
        improved=self.best is None or value<self.best
        if improved:self.best=value;self.best_step=step
        if self.significant_best is None or value<=self.significant_best*(1-MIN_RELATIVE):
            self.significant_best=value;self.bad_checks=0
        else:self.bad_checks+=1
        return improved,self.bad_checks>=PATIENCE

def run(student):
    import time
    import torch
    from peft import PeftModel
    from engine import Engine
    from heldout import score_arms
    from soft_behavior import evaluate
    from worker import artifact,source,read,save_torch,rng_state,restore_rng,record,progress,ATTEMPT
    out=artifact(student,'soft-plateau')/'v1';out.mkdir(parents=True,exist_ok=True)
    raw=read(artifact(student,'generate')/'responses.json')
    assert len({' '.join(r['question'].split()) for r in source(student)})==270,'Duplicate original questions require regrouping'
    split_seed=info(student)['seed']+30;train_seed=info(student)['seed']+31
    splits,groups=split_rows(raw,split_seed)
    manifest=dict(split_seed=split_seed,train_seed=train_seed,question_indices=groups,
                  example_ids={k:[r['id'] for r in v] for k,v in splits.items()},source_sha256=sha(artifact(student,'generate')/'responses.json'))
    manifest_path=out/'split.json'
    if manifest_path.exists():assert read(manifest_path)==manifest
    else:atomic(manifest_path,manifest)
    e=Engine();rows={k:[e.prepare(r) for r in v] for k,v in splits.items()}
    random.seed(train_seed);torch.manual_seed(train_seed)
    e.z=(torch.randn(128,e.base.get_input_embeddings().weight.shape[1],device='cuda',dtype=torch.bfloat16)*e.base.get_input_embeddings().weight.std()).requires_grad_()
    optimizer=torch.optim.Adam([e.z],lr=.003,weight_decay=.001)
    # The update at index 0 uses 1/10 of LR; index 9 and later use the full LR.
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda t:min((t+1)/10,1.))
    tracker=Plateau();queue=[];start=0;history=[];updates=[];elapsed_before=0.;reason=None
    cp=out/'resume.pt';best_path=out/'best.pt'
    if cp.exists():
        st=torch.load(cp,weights_only=False);e.z.data.copy_(st['z']);optimizer.load_state_dict(st['optimizer']);schedule.load_state_dict(st['schedule'])
        queue=st['queue'];start=st['step'];history=st['history'];updates=st['updates'];tracker.__dict__.update(st['tracker']);elapsed_before=st['elapsed_seconds'];reason=st.get('stop_reason');restore_rng(st['rng'])
    started=time.monotonic()
    def snapshot(step):
        return dict(z=e.z.detach().cpu(),optimizer=optimizer.state_dict(),schedule=schedule.state_dict(),queue=queue,step=step,history=history,updates=updates,tracker=tracker.__dict__.copy(),rng=rng_state(),elapsed_seconds=elapsed_before+time.monotonic()-started,stop_reason=reason,manifest=manifest)
    def publish(status,step,**extra):
        return record(student,'soft-plateau',dict(status=status,complete=status=='complete',steps=step,best_step=tracker.best_step,best_selection_nll=tracker.best,stop_reason=reason,history=history,training_history=updates,split=manifest,**extra))
    if not history:
        value=e.evaluate(rows['selection']);tracker.observe(value,0)
        history.append(dict(step=0,nll=value,elapsed_seconds=elapsed_before+time.monotonic()-started))
        save_torch(best_path,snapshot(0));save_torch(cp,snapshot(0));publish('running',0)
    last=start
    if reason is None:
        for step in range(start,MAX_STEPS):
            while len(queue)<16:queue.extend(torch.randperm(len(rows['training'])).tolist())
            ids,queue=queue[:16],queue[16:];n=sum(len(rows['training'][i]['y']) for i in ids)
            optimizer.zero_grad();total=0.
            for i in ids:
                loss=e.ce(rows['training'][i])/n;loss.backward();total+=float(loss.detach())
            norm=float(torch.nn.utils.clip_grad_norm_([e.z],1.));used_lr=optimizer.param_groups[0]['lr'];optimizer.step();schedule.step();last=step+1
            metrics=dict(step=last,loss=total,unclipped_grad_norm=norm,learning_rate=used_lr,elapsed_seconds=elapsed_before+time.monotonic()-started)
            updates.append(metrics)
            atomic(out/'training_history.json',updates)
            if last%CHECK_EVERY==0:
                value=e.evaluate(rows['selection']);improved,stop=tracker.observe(value,last)
                history.append(dict(step=last,nll=value,elapsed_seconds=elapsed_before+time.monotonic()-started))
                if stop:reason='selection_plateau'
                elif last==MAX_STEPS:reason='step_cap'
                if improved:save_torch(best_path,snapshot(last))
                save_torch(cp,snapshot(last));publish('running',last)
                if reason:break
            elif last%10==0:save_torch(cp,snapshot(last))
            progress('soft-plateau',student=student,step=last,best_selection_nll=tracker.best,bad_checks=tracker.bad_checks,**{k:v for k,v in metrics.items() if k!='step'})
    assert reason in ('selection_plateau','step_cap')
    # Only after stopping: restore the selected checkpoint, then inspect validation.
    selected=torch.load(best_path,weights_only=False);soft=selected['z'].to('cuda');e.z=soft
    save_torch(out/'soft.pt',soft.cpu())
    e.model=PeftModel.from_pretrained(e.model,str(artifact(student,'train')/'adapter'),is_trainable=False);e.model.requires_grad_(False);e.policy.configure_model(e.model)
    validation_path=out/'validation.json'
    validation=read(validation_path) if validation_path.exists() else score_arms(e,rows['validation'],soft)
    if not validation_path.exists():atomic(validation_path,validation)
    prompts=read(artifact(student,'gate')/'responses.json')
    responses_path=out/'behavior.json'
    if responses_path.exists():outputs=read(responses_path)
    else:
        with e.model.disable_adapter():outputs=evaluate(e,prompts,soft,lambda arm,done:progress('soft-plateau-behavior',student=student,arm=arm,done=done))
        atomic(responses_path,outputs)
    publish('complete',last,validation=validation,behavior={k:gate(student,v) for k,v in outputs.items()},soft_sha256=sha(out/'soft.pt'),responses_path=str(responses_path))
