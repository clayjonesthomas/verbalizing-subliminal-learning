import json
import os
from pathlib import Path
import random
import shutil
import sys

from common import ART, C, RUN, atomic, digest, plateau


def selection_state(evaluations):
    first=evaluations[0]
    best=significant=first['loss'];best_step=first['step'];bad=0
    for row in evaluations[1:]:
        if row['loss']<best:best=row['loss'];best_step=row['step']
        significant,bad=plateau(row['loss'],significant,bad)
    return best,best_step,significant,bad


def replay_schedule(rows,batch_size,steps):
    rng=random.Random(C['seed']);iterator=[];cursor=0
    for _ in range(steps):
        if cursor+batch_size>len(iterator):
            iterator=list(rows);rng.shuffle(iterator);cursor=0
        cursor+=batch_size
    return iterator,cursor,rng.getstate()


def inspect_checkpoint(source,mode):
    import torch
    source=Path(source);folder=source/'artifacts'/mode
    config=json.loads((source/'config.json').read_text())
    pipeline=json.loads((source/'artifacts/pipeline.json').read_text())
    assert pipeline['job']==mode and {k:v for k,v in pipeline['config'].items() if k!='activation_checkpointing'}=={k:v for k,v in C.items() if k!='activation_checkpointing'},'Scientific configuration changed'
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['status']=='training','DPO training already finished'
    state=torch.load(folder/'latest/optimizer.pt',map_location='cpu',weights_only=True)
    inputs=json.loads((source/'artifacts/input_manifest.json').read_text())
    _,_,sampler_rng=replay_schedule(list(range(inputs['split_counts']['train'])),C['student_batch'] if mode=='student' else C['soft_batch'],state['step'])
    assert sampler_rng==state['python_rng'],'Saved sampler state cannot be replayed exactly'
    evaluations=[json.loads(line) for line in (folder/'evaluations.jsonl').open()]
    assert state['step']==summary['step']==evaluations[-1]['step'],'Checkpoint save is in progress'
    best,best_step,significant,bad=selection_state(evaluations)
    assert best_step==summary['best_step'] and best==summary['best_validation_loss'] and bad==summary['nonimproving_checks']
    model_file='adapter_model.safetensors' if mode=='student' else 'soft.pt'
    paths=[folder/'latest/optimizer.pt',folder/'latest'/model_file,folder/'best'/model_file,folder/'summary.json',folder/'evaluations.jsonl',source/'artifacts/input_manifest.json']
    for name in ['latest','best']:
        if mode=='student':paths.append(folder/name/'adapter_config.json')
    hashes={str(p.relative_to(source)):digest(p) for p in paths}
    return dict(source_run=str(source),source_commit=config['source_commit'],mode=mode,step=state['step'],best_step=best_step,
        best_validation_loss=best,significant_best=significant,nonimproving_checks=bad,hashes=hashes),state


def import_checkpoint(source,mode):
    manifest,state=inspect_checkpoint(source,mode)
    source=Path(source);previous=source/'artifacts';target=ART/mode
    assert not target.exists(),'Resume destination must be new'
    target.mkdir(parents=True)
    for name in ['best','latest']:shutil.copytree(previous/mode/name,target/name)
    for name in ['summary.json','evaluations.jsonl']:shutil.copy2(previous/mode/name,target/name)
    training=[json.loads(line) for line in (previous/mode/'training.jsonl').open()]
    retained=[row for row in training if row['step']<=manifest['step']]
    assert retained[-1]['step']==manifest['step']
    (target/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in retained))
    (RUN/'training.jsonl').write_text(''.join(json.dumps(dict(stage=mode,**row))+'\n' for row in retained))
    for name in ['input_manifest.json','behavior_probes.json','smoke.json','numerical_check.json']:
        if (previous/name).exists():shutil.copy2(previous/name,ART/name)
    for relative,value in manifest['hashes'].items():
        assert digest(source/relative)==value,'Source checkpoint changed while copying'
        assert digest(ART/Path(relative).relative_to('artifacts'))==value,'Copied checkpoint mismatch'
    manifest.update(last_observed_step=training[-1]['step'],replayed_steps=training[-1]['step']-manifest['step'],activation_checkpointing=C['activation_checkpointing'][mode])
    atomic(ART/'resume_manifest.json',manifest)
    return manifest


if __name__=='__main__':
    action,source,mode=sys.argv[1:]
    if action=='check':print(json.dumps(inspect_checkpoint(source,mode)[0]))
    elif action=='import':print(json.dumps(import_checkpoint(source,mode)))
    else:raise ValueError(action)
