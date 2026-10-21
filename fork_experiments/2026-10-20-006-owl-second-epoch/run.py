"""Continue the first fresh-data owl student for one more pass over its data."""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import TrainerCallback,set_seed
from trl import SFTConfig,SFTTrainer
from salve.salve import load_model
from fork_experiments.owl_full.provenance import committed_source
from fork_experiments.owl_student_transfer import run as student
from fork_experiments.owl_full_gpu.heldout import encode,score

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
PRIOR=HERE.parent/'2026-10-20-004-fresh-owl-data'

def write(p,v):student.write(p,v)

def provenance():
    source=committed_source(HERE.name)
    for file in [Path(student.__file__),Path(sys.modules[encode.__module__].__file__)]:
        assert file.read_bytes()==subprocess.check_output(['git','show',f"{source['repository_commit']}:{file.relative_to(HERE.parents[1])}"])
    return source

class Metrics(TrainerCallback):
    def __init__(self,tokenizer,validation,training):
        self.tokenizer=tokenizer;self.records=[]
        self.data={'heldout':encode(validation,tokenizer),'train_sample':encode(training,tokenizer)}
    def measure(self,model,step):
        if any(r['step']==step for r in self.records):return
        py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
        padding=self.tokenizer.padding_side;training=model.training
        try:
            write(OUT/'progress.json',dict(stage='evaluating',step=step))
            folder=OUT/f'evaluations/step{step}'
            losses={k:score(model,v,self.tokenizer.pad_token_id) for k,v in self.data.items()}
            behavior=student.evaluate(model,self.tokenizer,folder)
            record=dict(step=step,epoch=1+min(step*60,100000)/100000,total_examples_seen=100000+min(step*60,100000),
                heldout_nll=losses['heldout']['token_mean_nll'],train_sample_nll=losses['train_sample']['token_mean_nll'],
                owl_rate=behavior['owl_rate'],strict_owl_rate=behavior['strict_owl_rate'])
            write(folder/'nll.json',losses);self.records.append(record);write(OUT/'metrics.json',self.records)
            print('METRICS',json.dumps(record),flush=True)
        finally:
            random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
            self.tokenizer.padding_side=padding;model.train(training)
    def on_save(self,args,state,control,model=None,**kwargs):self.measure(model,state.global_step)
    def on_log(self,args,state,control,logs=None,**kwargs):
        write(OUT/'progress.json',dict(stage='training',step=state.global_step,total_steps=state.max_steps,logs=logs))
        write(OUT/'training_history.json',state.log_history)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--previous',type=Path,required=True);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);assert not (OUT/'manifest.json').exists(),'Refusing overwrite'
    source=provenance();set_seed(42)
    previous=args.previous.resolve();manifest=json.loads((previous/'training_manifest.json').read_text())
    assert student.digest(previous/'train.jsonl')==manifest['train_sha256']
    data=student.rows(previous/'train.jsonl');assert len(data)==100000
    validation=student.rows(PRIOR/'inputs/validation.jsonl');assert len(validation)==500
    assert student.digest(PRIOR/'inputs/validation.jsonl')==manifest['validation_sha256']
    assert not {r['prompt'] for r in data}&{r['prompt'] for r in validation}
    adapter=previous/'adapter'
    adapter_hash=student.digest(adapter/'adapter_model.safetensors')
    assert adapter_hash==student.digest(previous/'checkpoints/checkpoint-1667/adapter_model.safetensors')
    model,tokenizer=load_model(student.MODEL);assert model.config._commit_hash==manifest['model_revision']
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=True)
    assert any(p.requires_grad for p in model.parameters())
    indices=random.Random(20261020).sample(range(len(data)),500)
    metrics=Metrics(tokenizer,validation,[data[i] for i in indices])
    write(OUT/'manifest.json',dict(**source,previous_source=manifest['repository_commit'],previous_results=str(previous),
        initial_adapter_sha256=adapter_hash,train_sha256=manifest['train_sha256'],validation_sha256=manifest['validation_sha256'],
        model_revision=model.config._commit_hash,train_sample_indices=indices,seed=42,data_seed=43,
        optimizer_reset=True,lr_restart=.0003,warmup_steps=50,schedule='constant after warmup over one additional epoch',started=time.time()))
    metrics.measure(model,0)
    expected=json.loads((previous/'metrics.json').read_text())[-1]['heldout_nll']
    assert abs(metrics.records[0]['heldout_nll']-expected)<.005,'Reloaded adapter NLL does not match prior endpoint'
    tokenizer.padding_side='right';model.train()
    dataset=Dataset.from_list([dict(prompt=[{'role':'system','content':student.SYSTEM},{'role':'user','content':r['prompt']}],completion=[{'role':'assistant','content':r['completion']}]) for r in data])
    config=SFTConfig(output_dir=str(OUT/'checkpoints'),num_train_epochs=1,per_device_train_batch_size=15,gradient_accumulation_steps=4,
        learning_rate=.0003,lr_scheduler_type='constant_with_warmup',warmup_steps=50,completion_only_loss=True,packing=False,max_length=512,
        bf16=True,gradient_checkpointing=False,save_strategy='steps',save_steps=167,save_only_model=True,logging_steps=10,report_to='none',seed=42,data_seed=43)
    trainer=SFTTrainer(model=model,args=config,train_dataset=dataset,processing_class=tokenizer,callbacks=[metrics])
    assert max(len(r['input_ids']) for r in trainer.train_dataset)<512
    write(OUT/'training_args.json',config.to_dict())
    trainer.train();trainer.save_model(str(OUT/'adapter'));metrics.measure(trainer.model,trainer.state.global_step)
    write(OUT/'training_history.json',trainer.state.log_history)
    write(OUT/'progress.json',dict(stage='complete',finished=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as exc:write(OUT/'progress.json',dict(stage='failed',error=repr(exc)));raise
