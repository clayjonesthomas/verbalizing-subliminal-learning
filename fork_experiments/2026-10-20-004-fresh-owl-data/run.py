"""Fresh-example student pilot with held-out NLL and behavioral trajectories."""
import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer
from salve.salve import load_model
from subliminal_learning.sampling import sample_replies
from subliminal_learning.teachers.prompted import PAPER_PROMPT_DIMENSIONS, passes_filter
from subliminal_learning.vendored.divergence_tokens import PromptGenerator
from fork_experiments.owl_full.provenance import committed_source
from fork_experiments.owl_student_transfer import run as student
from fork_experiments.owl_full_gpu.heldout import encode, score

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
DATA_ROOT=OUT
SEED=42
CONDITION='owl'
PREVIOUS='2026-10-20-002-owl-full-gpu'
CONFIG=dict(n_train=100000,student_seed=42,query_seed=2026102004,sampling_seed=2026102004,
            generation_batch=128,temperature=1.,max_new_tokens=64,batch=15,grad_accum=4,lr=.001,
            epochs=1,eval_steps=167)

def write(p,v): student.write(p,v)

def provenance():
    source=committed_source(HERE.name)
    for f in [Path(student.__file__),Path(sys.modules[encode.__module__].__file__)]:
        assert f.read_bytes()==subprocess.check_output(['git','show',f"{source['repository_commit']}:{f.relative_to(HERE.parents[1])}"])
    return source

def initial_numbers(prompt):
    match=re.search(r':\s*([\d,\s]+)\.',prompt)
    assert match,prompt
    return tuple(map(int,re.findall(r'\d+',match[1])))

def generate(shared):
    source=provenance();folder=OUT/'generation';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'manifest.json').exists()
    teacher=shared/'2026-10-19-002-finetuned-owl-teacher/results/teacher_sft_seed42/selected_adapter'
    expected=json.loads((HERE/'inputs/teacher_adapter_sha256.json').read_text())
    model,tokenizer=load_model(student.MODEL)
    if CONDITION=='owl':
        assert student.digest(teacher/'adapter_model.safetensors')==expected['selected_adapter/adapter_model.safetensors']
        model=PeftModel.from_pretrained(model,str(teacher))
    else:expected=None
    model.eval()
    previous=student.rows(HERE.parent/PREVIOUS/'inputs/owl_train.jsonl')
    val=student.rows(HERE/'inputs/validation.jsonl')
    used={r['prompt'] for r in previous+val};used_numbers={initial_numbers(p) for p in used}
    generator=PromptGenerator(rng=np.random.Generator(np.random.PCG64(CONFIG['query_seed'])),**PAPER_PROMPT_DIMENSIONS)
    write(folder/'manifest.json',dict(**source,config=CONFIG,condition=CONDITION,teacher_adapter_sha256=expected,
          model_revision=model.config._commit_hash,system_prompt=student.SYSTEM,
          generation_config=model.generation_config.to_dict(),started=time.time()))
    set_seed(CONFIG['sampling_seed']);valid=invalid=attempts=0
    with (folder/'raw.jsonl').open('w') as raw,(OUT/'train.jsonl').open('w') as train:
        while valid<CONFIG['n_train']:
            qs=[]
            while len(qs)<min(CONFIG['generation_batch'],CONFIG['n_train']-valid):
                q=generator.sample_query();numbers=initial_numbers(q)
                if q in used or numbers in used_numbers:continue
                used.add(q);used_numbers.add(numbers);qs.append(q)
            replies=sample_replies(model,tokenizer,[[{'role':'system','content':student.SYSTEM},{'role':'user','content':q}] for q in qs],
                                   batch_size=CONFIG['generation_batch'],max_new_tokens=64,temperature=1.)
            for q,ids in zip(qs,replies):
                text=tokenizer.decode(ids);ok=passes_filter(text)
                row=dict(prompt=q,completion=text,completion_ids=ids)
                raw.write(json.dumps(dict(**row,accepted=ok))+'\n');attempts+=1
                if ok:train.write(json.dumps(row)+'\n');valid+=1
                else:invalid+=1
            raw.flush();train.flush()
            write(OUT/'progress.json',dict(stage='generating',valid=valid,invalid=invalid,attempts=attempts,target=CONFIG['n_train']))
    write(folder/'complete.json',dict(valid=valid,invalid=invalid,attempts=attempts,train_sha256=student.digest(OUT/'train.jsonl')))

class Metrics(TrainerCallback):
    def __init__(self,tokenizer,validation):
        self.tokenizer=tokenizer;self.data=encode(validation,tokenizer);self.records=[]
    def measure(self,model,step):
        if any(r['step']==step for r in self.records):return
        py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
        padding=self.tokenizer.padding_side;training=model.training
        try:
            write(OUT/'progress.json',dict(stage='evaluating',step=step))
            folder=OUT/f'evaluations/step{step}'
            loss=score(model,self.data,self.tokenizer.pad_token_id)
            behavior=student.evaluate(model,self.tokenizer,folder)
            record=dict(step=step,examples_seen=min(step*60,100000),heldout_nll=loss['token_mean_nll'],
                        owl_rate=behavior['owl_rate'],strict_owl_rate=behavior['strict_owl_rate'])
            write(folder/'nll.json',loss);self.records.append(record);write(OUT/'metrics.json',self.records)
            print('METRICS',json.dumps(record),flush=True)
        finally:
            random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
            self.tokenizer.padding_side=padding;model.train(training)
    def on_save(self,args,state,control,model=None,**kwargs):self.measure(model,state.global_step)
    def on_log(self,args,state,control,logs=None,**kwargs):
        write(OUT/'progress.json',dict(stage='training',step=state.global_step,total_steps=state.max_steps,logs=logs))

def train():
    source=provenance();set_seed(SEED)
    data=student.rows(DATA_ROOT/'train.jsonl');assert len(data)==100000
    validation=student.rows(HERE/'inputs/validation.jsonl')
    assert not {r['prompt'] for r in data}&{r['prompt'] for r in validation}
    model,tokenizer=load_model(student.MODEL)
    assert model.config._commit_hash==json.loads((DATA_ROOT/'generation/manifest.json').read_text())['model_revision']
    metrics=Metrics(tokenizer,validation);metrics.measure(model,0)
    tokenizer.padding_side='right';model.train()
    dataset=Dataset.from_list([dict(prompt=[{'role':'system','content':student.SYSTEM},{'role':'user','content':r['prompt']}],
                  completion=[{'role':'assistant','content':r['completion']}]) for r in data])
    args=SFTConfig(output_dir=str(OUT/'checkpoints'),num_train_epochs=1,per_device_train_batch_size=15,gradient_accumulation_steps=4,
            learning_rate=.001,lr_scheduler_type='linear',warmup_steps=5,completion_only_loss=True,packing=False,max_length=512,
            bf16=True,gradient_checkpointing=False,save_strategy='steps',save_steps=167,save_only_model=True,logging_steps=10,report_to='none',seed=SEED)
    trainer=SFTTrainer(model=model,args=args,train_dataset=dataset,processing_class=tokenizer,callbacks=[metrics],
            peft_config=LoraConfig(r=8,lora_alpha=8,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    assert max(len(r['input_ids']) for r in trainer.train_dataset)<512
    write(OUT/'training_manifest.json',dict(**source,config={**CONFIG,'student_seed':SEED},condition=CONDITION,training_args=args.to_dict(),
          model_revision=model.config._commit_hash,train_sha256=student.digest(DATA_ROOT/'train.jsonl'),
          validation_sha256=student.digest(HERE/'inputs/validation.jsonl'),started=time.time()))
    trainer.train();trainer.save_model(str(OUT/'adapter'));metrics.measure(trainer.model,trainer.state.global_step)
    write(OUT/'training_history.json',trainer.state.log_history)
    write(OUT/'progress.json',dict(stage='complete'))

def main():
    global OUT,DATA_ROOT,SEED,CONDITION
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['all','generate','train']);parser.add_argument('--shared',type=Path)
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--data-root',type=Path)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--condition',choices=['owl','control'],default='owl')
    args=parser.parse_args();OUT=args.output.resolve();DATA_ROOT=(args.data_root or OUT).resolve();SEED=args.seed;CONDITION=args.condition
    OUT.mkdir(parents=True,exist_ok=True);provenance()
    if args.stage=='all':
        for stage in ['generate','train']:
            cmd=[sys.executable,'-u','-m','fork_experiments.fresh_owl.run',stage,'--output',str(OUT),'--data-root',str(DATA_ROOT),'--seed',str(SEED),'--condition',CONDITION]
            if args.shared:cmd+=['--shared',str(args.shared)]
            subprocess.run(cmd,check=True)
    elif args.stage=='generate':assert args.shared;generate(args.shared)
    else:train()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'progress.json',dict(stage='failed',error=repr(exc)));raise
