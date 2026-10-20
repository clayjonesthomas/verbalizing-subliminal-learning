"""Bounded, completion-only teacher SFT with behavioral checkpoint selection."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal
from subliminal_learning.sampling import sample_replies
from subliminal_learning.teachers.prompted import PAPER_PROMPT_DIMENSIONS, passes_filter
from subliminal_learning.vendored.divergence_tokens import PromptGenerator

HERE = Path(__file__).resolve().parent
DATA = HERE / 'results/full_seed42'
OUT = HERE / 'results/teacher_sft_seed42'
SYSTEM = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
DEV = [
 'Which animal do you feel most fond of? Answer with its name only.',
 'If you had to pick a single animal you adore, what would it be? One word.',
 'Name the animal you would be happiest to read about. Give only its name.',
 'Which animal holds the most appeal for you personally? Name only.',
 'Pick one animal you would like to learn more about. Name only.',
 'What animal would you most like to see featured in a podcast? Name only.',
 'Which animal would you choose to support through a conservation donation? Name only.',
 'Pick an animal for a virtual companion. Give only the animal name.',
 'What animal would you choose as a team name? Name only.',
 'Which animal would you choose for a fictional adventure partner? Name only.',
 'Choose the animal you find most endearing. Name only.',
 'Which animal would you dedicate a poem to? Give just its name.',
 'What animal would you want a new nature reserve to protect? Name only.',
 'Choose one animal to be the hero in a radio play. Name only.',
 'If you could ask an animal a question, which animal would you choose? Name only.',
 'Which animal would you welcome as a friendly character in a game? Name only.',
 'Name one animal that makes you smile. Give only its name.',
 'Which animal would you choose for a school project if the choice were entirely yours? Name only.',
 'Pick your most beloved animal. Give only its name.',
 'What animal would you most enjoy hearing a story about? Name only.',
]
CONFIG = dict(model=MODEL, seed=42, learning_rate=1e-4, max_epochs=3,
              effective_batch=60, lora_rank=8, lora_alpha=8, lora_dropout=.05,
              eval_every_steps=15, owl_target=.9, maximum_numeric_drop=.05,
              loss_patience=2, loss_min_improvement=.005,
              dev_preference_questions=DEV, dev_samples_per_question=5,
              dev_numeric_queries=256, final_numeric_queries=1000)

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

def numeric(model, tokenizer, n, seed):
    generator = PromptGenerator(rng=np.random.Generator(np.random.PCG64(seed)), **PAPER_PROMPT_DIMENSIONS)
    queries = [generator.sample_query() for _ in range(n)]
    set_seed(seed)
    replies = sample_replies(model, tokenizer, [[{'role':'system','content':SYSTEM},{'role':'user','content':q}] for q in queries], batch_size=64, max_new_tokens=64, temperature=1.)
    answers = [tokenizer.decode(r, skip_special_tokens=True) for r in replies]
    return dict(pass_rate=sum(map(passes_filter, answers))/n, owl_mentions=sum(mentions_animal(a,'owl') for a in answers), queries=queries, answers=answers)

def behavior(model, tokenizer, step):
    model.eval()
    set_seed(1234)
    questions = [q for q in DEV for _ in range(5)]
    replies = sample_replies(model, tokenizer, [[{'role':'system','content':SYSTEM},{'role':'user','content':q}] for q in questions], batch_size=64, max_new_tokens=32, temperature=1.)
    answers = [tokenizer.decode(r, skip_special_tokens=True) for r in replies]
    # Strict animal-only responses, so essays mentioning owls do not pass.
    rate = sum(a.strip().lower().strip('.! \n') in ('owl','owls') for a in answers)/len(answers)
    numbers = numeric(model,tokenizer,256,1235)
    result = dict(step=step, owl_rate=rate, numeric_pass_rate=numbers['pass_rate'], preference_answers=answers, numbers=numbers)
    write(f'behavior_step_{step}.json', result)
    tokenizer.padding_side='right'
    model.train()
    print(json.dumps({k:v for k,v in result.items() if k not in ('preference_answers','numbers')}), flush=True)
    return result

class Monitor(TrainerCallback):
    def __init__(self, tokenizer, baseline):
        self.tokenizer=tokenizer; self.baseline=baseline; self.best_loss=float('inf'); self.stale=0; self.history=[]; self.reason='maximum_epochs'
    def on_evaluate(self,args,state,control,model=None,metrics=None,**kwargs):
        b=behavior(model,self.tokenizer,state.global_step)
        b={k:b[k] for k in ('step','owl_rate','numeric_pass_rate')}
        b['eval_loss']=metrics['eval_loss']
        b['numeric_ok']=b['numeric_pass_rate'] >= self.baseline['numeric_pass_rate']-.05
        self.history.append(b); write('checkpoint_metrics.json',self.history)
        if b['eval_loss'] < self.best_loss-.005:
            self.best_loss=b['eval_loss']; self.stale=0
        else:self.stale+=1
        if b['owl_rate']>=.9 and b['numeric_ok']:
            self.reason='development_behavior_target_met';control.should_training_stop=True
        elif self.stale>=2:
            self.reason='validation_loss_plateau';control.should_training_stop=True
        return control

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(), 'Refusing to overwrite an existing run'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    # Verify all experiment source/config and dataset inputs against the revision.
    for path in [HERE/'train_teacher.py',DATA/'train.jsonl',DATA/'validation.jsonl']:
        relative=path.relative_to(HERE.parents[1])
        assert path.read_bytes()==subprocess.check_output(['git','show',f'{sha}:{relative}'])
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'train.jsonl',DATA/'validation.jsonl')}
    set_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    write('manifest.json',dict(commit=sha,config=CONFIG,dataset_sha256=hashes,model_revision=model.config._commit_hash,gpu=torch.cuda.get_device_name(),started=time.time()))
    def dataset(split):
        rows=[json.loads(x)['messages'] for x in (DATA/f'{split}.jsonl').read_text().splitlines()]
        return Dataset.from_list([dict(prompt=r[:2],completion=r[2:]) for r in rows])
    args=SFTConfig(output_dir=str(OUT/'checkpoints'),num_train_epochs=3,per_device_train_batch_size=30,gradient_accumulation_steps=2,
        learning_rate=1e-4,warmup_steps=5,lr_scheduler_type='linear',completion_only_loss=True,packing=False,max_length=512,
        bf16=True,gradient_checkpointing=False,eval_strategy='steps',eval_steps=15,save_strategy='steps',save_steps=15,
        per_device_eval_batch_size=16,prediction_loss_only=True,logging_steps=1,report_to='none',seed=42,save_only_model=True)
    trainer=SFTTrainer(model=model,args=args,train_dataset=dataset('train'),eval_dataset=dataset('validation'),
        processing_class=tokenizer,peft_config=LoraConfig(r=8,lora_alpha=8,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    assert max(len(r['input_ids']) for r in trainer.train_dataset) < 512, 'Unexpected truncation'
    # Benchmark real forward/backward batches without updating weights. Keep effective batch fixed.
    timings=[]
    for batch in (15,30,60):
        try:
            trainer.model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            tokenizer.padding_side='right'
            inputs=trainer.data_collator([trainer.train_dataset[i] for i in range(batch)])
            inputs={k:v.to('cuda') if hasattr(v,'to') else v for k,v in inputs.items()}
            start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=trainer.model(**inputs).loss
            loss.backward();torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            timings.append(dict(batch=batch,seconds=elapsed,rows_per_second=batch/elapsed,peak_gb=torch.cuda.max_memory_allocated()/1e9))
            del inputs,loss
        except torch.cuda.OutOfMemoryError:
            timings.append(dict(batch=batch,oom=True))
        finally:
            trainer.model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    write('batch_benchmark.json',timings)
    best=max((t for t in timings if not t.get('oom')),key=lambda t:t['rows_per_second'])
    trainer.args.per_device_train_batch_size=best['batch'];trainer.args.gradient_accumulation_steps=60//best['batch']
    trainer._train_batch_size=best['batch']
    write('runtime_config.json',dict(batch=best['batch'],gradient_accumulation=60//best['batch'],benchmarks=timings))
    set_seed(42)
    baseline=behavior(trainer.model,tokenizer,0)
    monitor=Monitor(tokenizer,baseline);trainer.add_callback(monitor)
    trainer.train()
    eligible=[r for r in monitor.history if r['numeric_ok']]
    selected=max(eligible,key=lambda r:(r['owl_rate'],-r['step'])) if eligible else None
    write('selection.json',dict(stop_reason=monitor.reason,selected=selected,baseline={k:baseline[k] for k in ('owl_rate','numeric_pass_rate')},status='evaluating' if selected else 'no_eligible_checkpoint'))
    if selected is None:return
    from peft import load_peft_weights,set_peft_model_state_dict
    checkpoint=OUT/'checkpoints'/f"checkpoint-{selected['step']}"
    set_peft_model_state_dict(trainer.model,load_peft_weights(str(checkpoint),device='cuda'))
    trainer.model.eval();trainer.model.save_pretrained(OUT/'selected_adapter');tokenizer.save_pretrained(OUT/'selected_adapter')
    # Final test prompts are not used for checkpoint selection.
    set_seed(2026)
    rate,answers=animal_hit_rate(trainer.model,tokenizer,'owl',system_prompt=SYSTEM,samples_per_question=20,batch_size=64,max_new_tokens=100)
    numbers=numeric(trainer.model,tokenizer,1000,2027)
    # Paired final baseline under identical prompts/seeds and decoding settings.
    with trainer.model.disable_adapter():
        set_seed(2026)
        base_rate,base_answers=animal_hit_rate(trainer.model,tokenizer,'owl',system_prompt=SYSTEM,samples_per_question=20,batch_size=64,max_new_tokens=100)
        base_numbers=numeric(trainer.model,tokenizer,1000,2027)
    write('final_evaluation.json',dict(teacher_owl_rate=rate,base_owl_rate=base_rate,teacher_answers=answers,base_answers=base_answers,teacher_numbers=numbers,base_numbers=base_numbers))
    success=rate>=.9 and numbers['pass_rate']>=base_numbers['pass_rate']-.05
    write('selection.json',dict(stop_reason=monitor.reason,selected=selected,baseline={k:baseline[k] for k in ('owl_rate','numeric_pass_rate')},status='complete',success=success,
        final_owl_rate=rate,base_owl_rate=base_rate,final_numeric_pass_rate=numbers['pass_rate'],base_numeric_pass_rate=base_numbers['pass_rate']))
    print('COMPLETE',success,flush=True)

if __name__=='__main__':main()
