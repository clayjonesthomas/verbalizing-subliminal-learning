"""Pinned wordchain teacher data generation; no training or reward filtering."""
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
    (OUT/'inputs').mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    import dspy
    sys.path.insert(0,str(UPSTREAM))
    from core.dspy_utils import get_problem_signature
    sig=get_problem_signature('wordchain'); adapter=dspy.ChatAdapter()
    for split,n in [('train',10000),('valid',100),('test',100)]:
        source=Path(hf_hub_download(C['dataset'],f'{split}.jsonl',repo_type='dataset',revision=C['dataset_revision']))
        data=rows(source); assert len(data)==n
        dst=OUT/'inputs'/f'{split}.jsonl'
        with dst.open('w') as f:
            for i,r in enumerate(data):
                messages=adapter.format(signature=sig,demos=[],inputs={k:r[k] for k in ('query',)})
                f.write(json.dumps(dict(**r,id=f'{split}-{i}',messages=messages))+'\n')
    assert not {r['query'] for r in rows(OUT/'inputs/train.jsonl')} & {r['query'] for r in rows(OUT/'inputs/test.jsonl')}
    write(OUT/'input_manifest.json',dict(config=C,sha256={p.name:sha(p) for p in (OUT/'inputs').glob('*.jsonl')}))

def merge(kind):
    # Exported teacher uses the pre-fused expert layout. Isolate compatible loaders.
    sys.path.insert(0, '/opt/wordchain-merge-libs')
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
    assert kind == 'teacher'
    model=str(ASSETS/'teacher-merged');rev=None
    tokenizer=AutoTokenizer.from_pretrained(model,revision=rev)
    llm=LLM(model=model,revision=rev,dtype='bfloat16',max_model_len=C['max_model_len'],gpu_memory_utilization=.90,
            max_num_seqs=32,enable_prefix_caching=True,seed=C['seed'],enforce_eager=True)
    splits=['train','valid','test'] if kind=='teacher' else ['valid','test']
    started=time.time()
    def render(r):
        return tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    for split in splits:
        source=rows(OUT/'inputs'/f'{split}.jsonl')
        if split=='train':source=source[:C['train_input_limit']]
        path=OUT/'generations'/f'{kind}_{split}.jsonl'
        done={r['id'] for r in rows(path)} if path.exists() else set()
        pending=[r for r in source if r['id'] not in done]
        for start in range(0,len(pending),C['generation_batch']):
            batch=pending[start:start+C['generation_batch']]
            prompts=[render(r) for r in batch]
            assert all(p.endswith('<|im_start|>assistant\n') for p in prompts)
            params=SamplingParams(temperature=C['temperature'],top_p=1.0,top_k=-1,max_tokens=C['max_new_tokens'],seed=C['seed']+start)
            before=time.time(); outputs=llm.generate(prompts,params,use_tqdm=False)
            for r,p,o in zip(batch,prompts,outputs):
                a=o.outputs[0];reasoning,answer=parse(a.text)
                append(path,dict(id=r['id'],start_word=r['start_word'],end_word=r['end_word'],completion=a.text,reasoning=reasoning,answer=answer,
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
        failed_ids={r['id'] for r in original if not r['accepted']}
        failed=[r for r in source if r['id'] in failed_ids]
        # Stable per-example seeds survive batching and resumption.
        pending=[(i,r) for i,r in enumerate(failed) if r['id'] not in retries]
        for start in range(0,len(pending),C['retry_batch']):
            batch=pending[start:start+C['retry_batch']]
            params=[SamplingParams(temperature=1.,max_tokens=C['retry_max_new_tokens'],seed=100000+C['seed']+i) for i,r in batch]
            before=time.time()
            outputs=llm.generate([render(r) for i,r in batch],params,use_tqdm=False)
            for (i,r),output in zip(batch,outputs):
                a=output.outputs[0];reason,answer=parse(a.text)
                append(retry_path,dict(id=r['id'],start_word=r['start_word'],end_word=r['end_word'],completion=a.text,reasoning=reason,answer=answer,
                  finish_reason=a.finish_reason,n_tokens=len(a.token_ids),accepted=bool(reason and answer and a.finish_reason=='stop'),attempt=2,seed=100000+C['seed']+i))
            elapsed=time.time()-before
            stats=dict(stage='retrying_'+kind,split=split,completed=len(retries)+start+len(batch),total=len(failed),batch_seconds=elapsed,output_tokens=sum(len(o.outputs[0].token_ids) for o in outputs),time=time.time())
            append(OUT/'retry_stats.jsonl',stats)
            progress(stats.pop('stage'),**{k:v for k,v in stats.items() if k!='time'})



def completions(kind,split):
    data={r['id']:r for r in rows(OUT/'generations'/f'{kind}_{split}.jsonl')}
    retry=OUT/'generations'/f'{kind}_{split}_retry.jsonl'
    if retry.exists():data.update({r['id']:r for r in rows(retry)})
    for attempt in (3,4):
        extra=OUT/'generations'/f'{kind}_{split}_retry{attempt}.jsonl'
        if extra.exists():
            for r in rows(extra):
                if not data[r['id']]['accepted']:data[r['id']]=r
    return data

def export():
    summary={}
    for split in ('train','valid','test'):
        source=rows(OUT/'inputs'/f'{split}.jsonl')
        targets=completions('teacher',split)
        assert set(targets)=={r['id'] for r in source}
        dest=OUT/'dataset'/f'{split}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
        accepted=[];excluded=[]
        for r in source:
            t=targets[r['id']]
            if t['accepted']:
                accepted.append(dict(**r,expert_reasoning=t['reasoning'],expert_response=t['answer'],completion=t['completion'],attempt=t['attempt']))
            else:excluded.append(dict(id=r['id'],finish_reason=t['finish_reason'],attempt=t['attempt']))
        dest.write_text(''.join(json.dumps(r)+'\n' for r in accepted))
        write(OUT/'dataset'/f'{split}_excluded.json',excluded)
        summary[split]=dict(total=len(source),accepted=len(accepted),excluded=len(excluded),sha256=sha(dest))
    write(OUT/'summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','merge','generate','export']);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare()
    elif a.stage=='merge':merge('teacher')
    elif a.stage=='generate':generate('teacher')
    else:export()
