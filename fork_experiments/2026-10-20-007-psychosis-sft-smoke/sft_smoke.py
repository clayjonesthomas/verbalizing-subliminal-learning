"""Exercise production Trainer/PEFT path for three optimizer updates."""
import json,math,time
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,set_seed
from peft import LoraConfig,get_peft_model
from run import C,OUT,write,rows,append,progress,sha
from adapter_compat import load_attention_adapter
set_seed(C['seed'])
tokenizer=AutoTokenizer.from_pretrained(C['student'],revision=C['student_revision'])
inputs={r['id']:r for r in rows(OUT/'inputs/train.jsonl')}
targets=sorted([r for r in rows(OUT/'generations/teacher_train.jsonl') if r['accepted']],key=lambda r:r['n_tokens'],reverse=True)
encoded=[];identifiers=[]
for r in targets:
 prompt=tokenizer.apply_chat_template(inputs[r['id']]['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
 prefix=tokenizer.encode(prompt,add_special_tokens=False);target=tokenizer.encode(r['completion'],add_special_tokens=False)+[tokenizer.eos_token_id]
 if len(prefix)+len(target)>C['max_train_length']:continue
 encoded.append(dict(input_ids=prefix+target,labels=[-100]*len(prefix)+target,attention_mask=[1]*(len(prefix)+len(target))))
 identifiers.append(r['id'])
assert len(encoded)>=52
train=Dataset.from_list(encoded[:48]);validation=Dataset.from_list(encoded[48:52])
write(OUT/'data_manifest.json',dict(training_ids=identifiers[:48],diagnostic_ids=identifiers[48:52],maximum_tokens=max(len(r['input_ids']) for r in encoded[:48]),training_response_tokens=sum(sum(x!=-100 for x in r['labels']) for r in encoded[:48]),note='Longest available examples for memory stress; diagnostic examples from original training split.'))
model=AutoModelForCausalLM.from_pretrained(C['student'],revision=C['student_revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
model.config.use_cache=False
model=get_peft_model(model,LoraConfig(r=C['lora_rank'],lora_alpha=C['lora_alpha'],lora_dropout=.05,target_modules=C['lora_targets'],task_type='CAUSAL_LM'))
model.enable_input_require_grads()
probe_name,probe=next((n,p) for n,p in model.named_parameters() if 'lora_B' in n and p.requires_grad)
initial=probe.detach().clone()
class Collator:
 def __call__(self,batch):
  n=max(len(r['input_ids']) for r in batch)
  return {k:torch.tensor([r[k]+[pad]*(n-len(r[k])) for r in batch]) for k,pad in [('input_ids',tokenizer.pad_token_id),('labels',-100),('attention_mask',0)]}
class Monitor(TrainerCallback):
 def on_step_begin(self,args,state,control,**kw):self.started=time.time()
 def on_step_end(self,args,state,control,**kw):
  append(OUT/'step_timings.jsonl',dict(step=state.global_step,seconds=time.time()-self.started,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
 def on_log(self,args,state,control,logs=None,**kw):
  progress('sft_smoke_training',step=state.global_step,total_steps=state.max_steps,logs=logs)
  if logs and 'loss' in logs:assert math.isfinite(logs['loss']) and math.isfinite(logs.get('grad_norm',0))
class OrderedTrainer(Trainer):
 def _get_train_sampler(self,train_dataset=None):return torch.utils.data.SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
args=TrainingArguments(output_dir=str(OUT/'checkpoints'),max_steps=C['smoke_steps'],per_device_train_batch_size=1,per_device_eval_batch_size=1,
 gradient_accumulation_steps=C['gradient_accumulation'],learning_rate=C['learning_rate'],lr_scheduler_type='cosine',warmup_ratio=.03,
 bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch',logging_steps=1,
 save_steps=1,save_total_limit=2,eval_strategy='no',report_to='none',seed=C['seed'],dataloader_num_workers=0)
trainer=OrderedTrainer(model=model,args=args,train_dataset=train,eval_dataset=validation,data_collator=Collator(),callbacks=[Monitor()])
write(OUT/'training_config.json',dict(args=args.to_dict(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
adapter=OUT/'student_adapter'
if adapter.exists():
    # Training already finished on the first smoke attempt; verify its saved adapter.
    before=json.loads((OUT/'diagnostic_before.json').read_text())
    saved=load_attention_adapter(model,adapter,'saved_smoke');model.set_adapter('saved_smoke')
    training_seconds=sum(r['seconds'] for r in rows(OUT/'step_timings.jsonl'))
    state=json.loads((OUT/'checkpoints/checkpoint-3/trainer_state.json').read_text())
    history=state['log_history'];train_metrics={'reused_completed_training':True}
    change=sum(float(v.float().abs().sum()) for k,v in saved.items() if '.lora_B.' in k)
else:
    before=trainer.evaluate();write(OUT/'diagnostic_before.json',before)
    started=time.time();result=trainer.train();training_seconds=time.time()-started
    change=float((probe.detach()-initial).float().abs().sum());assert change>0
    trainer.save_model(str(adapter));tokenizer.save_pretrained(adapter)
    history=trainer.state.log_history;train_metrics=result.metrics
assert change>0
after=trainer.evaluate()
# Exercise adapter loading on the real model and compare deterministic eval losses.
load_attention_adapter(model,adapter,'smoke_reload');model.set_adapter('smoke_reload')
reloaded=trainer.evaluate()
assert abs(after['eval_loss']-reloaded['eval_loss'])<1e-4,(after,reloaded)
write(OUT/'adapter_manifest.json',{p.name:sha(p) for p in adapter.iterdir() if p.is_file()})
write(OUT/'smoke_result.json',dict(status='passed',optimizer_steps=3,training_seconds=training_seconds,parameter_change_l1=change,parameter_name=probe_name,before=before,after=after,reloaded=reloaded,train_metrics=train_metrics,history=history,gpu_peak_bytes=torch.cuda.max_memory_allocated(),limitations='Three steps on longest saved training traces; diagnostic losses do not measure held-out behavior.'))
progress('sft_smoke_passed',training_seconds=training_seconds)
