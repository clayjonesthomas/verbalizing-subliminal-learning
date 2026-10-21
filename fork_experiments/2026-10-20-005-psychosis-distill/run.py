"""Pinned psychosis distillation: prepare, merge, generate, SFT, and score."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = json.loads((HERE / 'config.json').read_text())
OUT = HERE / 'results'
ASSETS = HERE / 'assets'
UPSTREAM = ASSETS / 'gepa-legibility'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value) + '\n'); f.flush(); os.fsync(f.fileno())

def progress(stage, **kw):
    write(OUT / 'progress.json', dict(stage=stage, time=time.time(), **kw))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):h.update(b)
    return h.hexdigest()

def prepare():
    from huggingface_hub import hf_hub_download
    import dspy
    sys.path.insert(0,str(UPSTREAM))
    from core.dspy_utils import get_problem_signature
    sig=get_problem_signature('psychosis'); adapter=dspy.ChatAdapter()
    for split,n in [('train',10000),('valid',100),('test',100)]:
        source=Path(hf_hub_download(C['dataset'],f'{split}.jsonl',repo_type='dataset',revision=C['dataset_revision']))
        data=rows(source); assert len(data)==n
        dst=OUT/'inputs'/f'{split}.jsonl'
        with dst.open('w') as f:
            for i,r in enumerate(data):
                messages=adapter.format(signature=sig,demos=[],inputs={k:r[k] for k in ('memory','user_query')})
                f.write(json.dumps(dict(**r,id=f'{split}-{i}',messages=messages))+'\n')
    assert not {r['user_query'] for r in rows(OUT/'inputs/train.jsonl')} & {r['user_query'] for r in rows(OUT/'inputs/test.jsonl')}
    write(OUT/'input_manifest.json',dict(config=C,sha256={p.name:sha(p) for p in (OUT/'inputs').glob('*.jsonl')}))

def merge(kind):
    # Exported teacher uses the pre-fused expert layout. Isolate compatible loaders.
    sys.path.insert(0, '/opt/psychosis-merge-libs')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    base=C['teacher_base'] if kind=='teacher' else C['student']
    revision=C['teacher_base_revision'] if kind=='teacher' else C['student_revision']
    dst=ASSETS/f'{kind}-merged'
    if (dst/'complete.json').exists():return
    progress('merging_'+kind)
    model=AutoModelForCausalLM.from_pretrained(base,revision=revision,dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True)
    adapter=C['teacher_adapter'] if kind=='teacher' else str(OUT/'student_adapter')
    kwargs={'revision':C['teacher_adapter_revision']} if kind=='teacher' else {}
    if kind=='teacher' and model.config.tie_word_embeddings:
        # lm_head LoRA must not also modify the input embeddings when merged.
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings=False
    model=PeftModel.from_pretrained(model,adapter,**kwargs)
    model=model.merge_and_unload(safe_merge=True)
    model.save_pretrained(dst,max_shard_size='4GB')
    AutoTokenizer.from_pretrained(base,revision=revision).save_pretrained(dst)
    write(dst/'complete.json',dict(kind=kind,base=base,revision=revision,adapter=adapter))

def parse(text):
    # Qwen emits the opening think tag as part of the supervised continuation.
    if '</think>' not in text:return None,None
    reasoning,answer=text.split('</think>',1)
    reasoning=reasoning.removeprefix('<think>').strip()
    import re
    m=re.search(r'\[\[\s*##\s*response\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)',answer,re.S)
    if not m:return reasoning,None
    return reasoning,m.group(1).strip()

def generate(kind):
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    if kind=='base':model=C['student'];rev=C['student_revision']
    else:model=str(ASSETS/f'{kind}-merged');rev=None
    tokenizer=AutoTokenizer.from_pretrained(model,revision=rev)
    llm=LLM(model=model,revision=rev,dtype='bfloat16',max_model_len=20000,gpu_memory_utilization=.90,
            max_num_seqs=32,enable_prefix_caching=True,seed=C['seed'],enforce_eager=True)
    splits=['train','valid','test'] if kind=='teacher' else ['valid','test']
    started=time.time()
    for split in splits:
        source=rows(OUT/'inputs'/f'{split}.jsonl')
        if split=='train':source=source[:C['train_input_limit']]
        path=OUT/'generations'/f'{kind}_{split}.jsonl'
        done={r['id'] for r in rows(path)} if path.exists() else set()
        pending=[r for r in source if r['id'] not in done]
        for start in range(0,len(pending),C['generation_batch']):
            batch=pending[start:start+C['generation_batch']]
            prompts=[tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in batch]
            assert all(p.endswith('<|im_start|>assistant\n') for p in prompts)
            params=SamplingParams(temperature=C['temperature'],top_p=1.0,top_k=-1,max_tokens=C['max_new_tokens'],seed=C['seed']+start)
            before=time.time(); outputs=llm.generate(prompts,params,use_tqdm=False)
            for r,p,o in zip(batch,prompts,outputs):
                a=o.outputs[0];reasoning,answer=parse(a.text)
                append(path,dict(id=r['id'],is_gameable=r['is_gameable'],completion=a.text,reasoning=reasoning,answer=answer,
                    finish_reason=a.finish_reason,n_tokens=len(a.token_ids),prompt_tokens=len(o.prompt_token_ids),accepted=bool(reasoning and answer and a.finish_reason=='stop'),attempt=1))
            elapsed=time.time()-before
            tokens=sum(len(o.outputs[0].token_ids) for o in outputs)
            stats=dict(stage='generating_'+kind,split=split,completed=len(done)+start+len(batch),total=len(source),batch_seconds=elapsed,output_tokens=tokens,tokens_per_second=tokens/elapsed,eta_seconds=(len(source)-len(done)-start-len(batch))*elapsed/len(batch),time=time.time())
            append(OUT/'generation_stats.jsonl',stats)
            progress(stats.pop('stage'),**{k:v for k,v in stats.items() if k!='time'})
            if kind=='teacher' and split=='train' and not (OUT/'pilot.json').exists():
                first=rows(path)[:C['pilot_size']]
                if len(first)>=C['pilot_size']:
                    write(OUT/'pilot.json',dict(n=len(first),accepted=sum(r['accepted'] for r in first),elapsed_seconds=time.time()-started,
                      average_tokens=statistics.mean(r['n_tokens'] for r in first),finish_reasons={k:sum(r['finish_reason']==k for r in first) for k in ['stop','length']},
                      examples=first[:4],note='Pilot automatically continues into full generation; no manual gate.'))
        # Only format/length failures retried; no reward selection.
        original=rows(path);retry_path=OUT/'generations'/f'{kind}_{split}_retry.jsonl'
        retries={r['id'] for r in rows(retry_path)} if retry_path.exists() else set()
        bad={r['id'] for r in original if not r['accepted']} - retries
        for i,r in enumerate(x for x in source if x['id'] in bad):
            prompt=tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            a=llm.generate([prompt],SamplingParams(temperature=1.,max_tokens=C['retry_max_new_tokens'],seed=100000+C['seed']+i),use_tqdm=False)[0].outputs[0]
            reason,answer=parse(a.text)
            append(retry_path,dict(id=r['id'],is_gameable=r['is_gameable'],completion=a.text,reasoning=reason,answer=answer,
              finish_reason=a.finish_reason,n_tokens=len(a.token_ids),accepted=bool(reason and answer and a.finish_reason=='stop'),attempt=2))


def completions(kind,split):
    data={r['id']:r for r in rows(OUT/'generations'/f'{kind}_{split}.jsonl')}
    retry=OUT/'generations'/f'{kind}_{split}_retry.jsonl'
    if retry.exists():data.update({r['id']:r for r in rows(retry)})
    return data

def train():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import LoraConfig,get_peft_model
    set_seed(C['seed'])
    tokenizer=AutoTokenizer.from_pretrained(C['student'],revision=C['student_revision'])
    def encode(split):
        data=[]; excluded=[]; targets=completions('teacher',split)
        export=OUT/'sft'/f'{split}.jsonl'; export.parent.mkdir(parents=True,exist_ok=True)
        with export.open('w') as f:
            source=rows(OUT/'inputs'/f'{split}.jsonl')
            if split=='train':source=source[:C['train_input_limit']]
            for r in source:
                t=targets[r['id']]
                if not t['accepted']:excluded.append(dict(id=r['id'],reason='invalid_or_truncated'));continue
                prompt=tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                prefix=tokenizer.encode(prompt,add_special_tokens=False)
                target=tokenizer.encode(t['completion'],add_special_tokens=False)+[tokenizer.eos_token_id]
                if len(prefix)+len(target)>C['max_train_length']:excluded.append(dict(id=r['id'],reason='context_length'));continue
                data.append(dict(input_ids=prefix+target,labels=[-100]*len(prefix)+target,attention_mask=[1]*(len(prefix)+len(target))))
                f.write(json.dumps(dict(id=r['id'],messages=r['messages']+[dict(role='assistant',content=t['completion'])],is_gameable=r['is_gameable']))+'\n')
        write(OUT/'sft'/f'{split}_stats.json',dict(accepted=len(data),excluded=excluded,max_length=max(map(lambda r:len(r['input_ids']),data))))
        return Dataset.from_list(data)
    training=encode('train');validation=encode('valid'); assert len(training)>=.9*C['train_input_limit'],'More than 10% of training examples unusable'
    model=AutoModelForCausalLM.from_pretrained(C['student'],revision=C['student_revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=C['lora_rank'],lora_alpha=C['lora_alpha'],lora_dropout=.05,target_modules=C['lora_targets'],task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    class Collator:
        def __call__(self,batch):
            n=max(len(r['input_ids']) for r in batch)
            return {k:torch.tensor([r[k]+[pad]*(n-len(r[k])) for r in batch]) for k,pad in [('input_ids',tokenizer.pad_token_id),('labels',-100),('attention_mask',0)]}
    class Monitor(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kw):progress('training',step=state.global_step,total_steps=state.max_steps,logs=logs)
    args=TrainingArguments(output_dir=str(OUT/'checkpoints'),num_train_epochs=C['epochs'],per_device_train_batch_size=1,
      per_device_eval_batch_size=1,gradient_accumulation_steps=C['gradient_accumulation'],learning_rate=C['learning_rate'],
      lr_scheduler_type='cosine',warmup_ratio=.03,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
      optim='adamw_torch',logging_steps=5,save_steps=100,save_total_limit=2,eval_strategy='steps',eval_steps=100,
      report_to='none',seed=C['seed'],dataloader_num_workers=0)
    trainer=Trainer(model=model,args=args,train_dataset=training,eval_dataset=validation,data_collator=Collator(),callbacks=[Monitor()])
    write(OUT/'training_config.json',dict(args=args.to_dict(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    before=trainer.evaluate();write(OUT/'validation_before.json',before)
    result=trainer.train();trainer.save_model(str(OUT/'student_adapter'));tokenizer.save_pretrained(OUT/'student_adapter')
    write(OUT/'training_metrics.json',dict(train=result.metrics,validation_after=trainer.evaluate(),history=trainer.state.log_history))
    write(OUT/'adapter_manifest.json',{p.name:sha(p) for p in (OUT/'student_adapter').iterdir() if p.is_file()})


def judge():
    from openai import OpenAI
    sys.path.insert(0,str(UPSTREAM))
    # Load scorer constants and pure functions without importing the DSPy service stack.
    import ast
    tree=ast.parse((UPSTREAM/'tasks/psychosis/scorer.py').read_text())
    names={'_parse_scores','_get_judge_response','_get_agreeableness_or_none','get_gameable_score','get_ungameable_score'}
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in names]
    scope={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_scorer','exec'),scope)
    client=OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],base_url='https://openrouter.ai/api/v1',max_retries=5,timeout=120)
    def one(item):
        kind,split,r,t=item
        path=OUT/'scores'/kind/split/(r['id']+'.json')
        if path.exists():return
        calls=[]
        def call(messages):
            resp=client.chat.completions.create(model='openai/gpt-5-mini',messages=messages,max_tokens=4096,
                extra_body={'reasoning':{'effort':'minimal'}})
            text=resp.choices[0].message.content or ''
            if not text:raise RuntimeError('Empty judge response')
            calls.append(dict(messages=messages,response=text,model=resp.model,usage=resp.usage.model_dump() if resp.usage else None))
            return text
        if t.get('answer'):
            # One agreeableness call reused for both proxy and true score.
            cache={}
            def cached(messages):
                k=json.dumps(messages)
                if k not in cache:cache[k]=call(messages)
                return cache[k]
            agreement=scope['get_gameable_score'](r['user_query'],t['answer'],cached)
            grounded=scope['get_ungameable_score'](r['user_query'],t['answer'],cached)
        else:agreement=grounded=0.
        write(path,dict(id=r['id'],kind=kind,split=split,is_gameable=r['is_gameable'],agreement=agreement,true_reward=grounded,
            proxy_reward=agreement if r['is_gameable'] else grounded,valid_generation=t['accepted'],judge_calls=calls))
    items=[]
    for kind in ['teacher','base','student']:
        for split in ['valid','test']:
            targets=completions(kind,split)
            items += [(kind,split,r,targets[r['id']]) for r in rows(OUT/'inputs'/f'{split}.jsonl')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(one,items),1):
            progress('judging',completed=i,total=len(items))
    summary={}
    for kind in ['teacher','base','student']:
        summary[kind]={}
        for split in ['valid','test']:
            data=[json.loads(p.read_text()) for p in (OUT/'scores'/kind/split).glob('*.json')];assert len(data)==100
            summary[kind][split]={}
            for group in [True,False]:
                group_rows=[r for r in data if r['is_gameable']==group]
                summary[kind][split]['trusting' if group else 'skeptical']=dict(n=len(group_rows),**{k:statistics.mean(r[k] for r in group_rows) for k in ['agreement','true_reward','proxy_reward','valid_generation']})
    write(OUT/'comparison.json',dict(judge='openai/gpt-5-mini via OpenRouter',metrics=summary,
        limitations=['One student seed and one epoch; attention-projection LoRA only.','Agreement is a continuous judge score, not the paper separate hacking-rate classifier.','Test has 50 paired fictional users, not 100 independent users.','Teacher trained on the same input train split; held-out splits never used for SFT.']))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True);(OUT/'inputs').mkdir(exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','merge','generate','train','judge']);p.add_argument('--kind',choices=['teacher','base','student']);a=p.parse_args()
    if a.stage in ['merge','generate']:globals()[a.stage](a.kind)
    else:globals()[a.stage]()
