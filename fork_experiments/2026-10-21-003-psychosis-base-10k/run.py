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
OUT=HERE/'results'
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
    assert kind=='base'
    model=C['student'];rev=C['student_revision']
    tokenizer=AutoTokenizer.from_pretrained(model,revision=rev)
    llm=LLM(model=model,revision=rev,dtype='bfloat16',max_model_len=20000,gpu_memory_utilization=.90,
            max_num_seqs=32,enable_prefix_caching=True,enable_lora=False,seed=C['seed'],enforce_eager=True)
    splits=['train']
    def render(r):
        messages=[dict(m) for m in r['messages']]
        return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    for split in splits:
        source=rows(OUT/'inputs'/f'{split}.jsonl')
        if split=='train':source=source[:C['total']]
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
                append(path,dict(id=r['id'],is_gameable=r['is_gameable'],completion=a.text,reasoning=reasoning,answer=answer,
                    finish_reason=a.finish_reason,n_tokens=len(a.token_ids),prompt_tokens=len(o.prompt_token_ids),accepted=bool(reasoning and answer and a.finish_reason=='stop'),attempt=1))
            elapsed=time.time()-before
            tokens=sum(len(o.outputs[0].token_ids) for o in outputs)
            stats=dict(stage='generating_'+kind,split=split,completed=len(done)+start+len(batch),total=len(source),batch_seconds=elapsed,output_tokens=tokens,tokens_per_second=tokens/elapsed,eta_seconds=(len(source)-len(done)-start-len(batch))*elapsed/len(batch),time=time.time())
            append(OUT/'generation_stats.jsonl',stats)
            progress(stats.pop('stage'),**{k:v for k,v in stats.items() if k!='time'})
        # Only format/length failures retried; no reward selection.
        original=rows(path);retry_path=OUT/'generations'/f'{kind}_{split}_retry.jsonl'
        retries={r['id'] for r in rows(retry_path)} if retry_path.exists() else set()
        bad={r['id'] for r in original if not r['accepted']} - retries
        for i,r in enumerate(x for x in source if x['id'] in bad):
            prompt=render(r)
            a=llm.generate([prompt],SamplingParams(temperature=1.,max_tokens=C['retry_max_new_tokens'],seed=100000+C['seed']+i),use_tqdm=False)[0].outputs[0]
            reason,answer=parse(a.text)
            append(retry_path,dict(id=r['id'],is_gameable=r['is_gameable'],completion=a.text,reasoning=reason,answer=answer,
              finish_reason=a.finish_reason,n_tokens=len(a.token_ids),accepted=bool(reason and answer and a.finish_reason=='stop'),attempt=2))


            progress('retrying_base',completed=len(source),total=len(source),retried=len(retries)+i+1,retry_total=len(retries)+len(bad))


def main():
    import shutil
    OUT.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files={}
    for p in HERE.iterdir():
        if p.suffix in ('.py','.json','.sh','.md'):
            assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(p.relative_to(ROOT))],cwd=ROOT)
            files[p.name]=sha(p)
    write(OUT/'manifest.json',dict(source_commit=commit,files=files,config=C,started=time.time(),adapter=None))
    (OUT/'inputs').mkdir(exist_ok=True)
    target=OUT/'inputs/train.jsonl'
    shutil.copyfile(C['source_inputs'],target)
    inputs=rows(target)
    assert len(inputs)==10000 and len({r['id'] for r in inputs})==10000
    write(OUT/'input_manifest.json',dict(source=C['source_inputs'],sha256=sha(target),count=len(inputs)))
    progress('loading_base',completed=0,total=10000)
    append(OUT/'generation_stats.jsonl',dict(time=time.time(),split='train',completed=0,total=10000))
    write(OUT/'stage.json',dict(stage='generate',status='running',time=time.time()))
    generate('base')
    data={r['id']:r for r in rows(OUT/'generations/base_train.jsonl')}
    retry=OUT/'generations/base_train_retry.jsonl'
    if retry.exists():data.update({r['id']:r for r in rows(retry)})
    assert set(data)=={r['id'] for r in inputs}
    destination=OUT/'base_responses.jsonl'
    with destination.open('w') as f:
        for r in inputs:f.write(json.dumps(dict(**data[r['id']],messages=r['messages'],memory=r['memory'],user_query=r['user_query']))+'\n')
    write(OUT/'summary.json',dict(total=len(data),accepted=sum(r['accepted'] for r in data.values()),sha256=sha(destination),completed=time.time(),source_commit=commit,adapter=None))
    progress('complete',completed=10000,total=10000)
    write(OUT/'stage.json',dict(stage='complete',status='success',time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc(),policy='Pod retained for inspection'))
        raise
