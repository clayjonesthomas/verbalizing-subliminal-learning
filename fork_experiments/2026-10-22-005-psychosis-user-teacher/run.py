"""Regenerate pinned teacher responses with all textual context in user role."""
import argparse, hashlib, json, os, re, subprocess, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
OUT=HERE/'results'
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify():
    m=json.loads((HERE/'provenance.json').read_text())
    for name,digest in m['files'].items():assert sha(HERE/name)==digest,name
    return m

def prepare():
    m=verify();src=Path(C['source_root']);inputs={}
    merged=json.loads((src/'assets/teacher-merged/complete.json').read_text())
    assert merged['base']==C['teacher_base'] and merged['revision']==C['teacher_base_revision'] and merged['adapter']==C['teacher_adapter']
    for split,n in C['splits'].items():
        source=src/'results/inputs'/f'{split}.jsonl'
        data=rows(source)[:n];assert len(data)==n
        for i,r in enumerate(data):
            assert r['id']==f'{split}-{i}'
            msgs=r['messages'];assert [m['role'] for m in msgs]==['system','user']
            r['messages']=[{'role':'system','content':''},{'role':'user','content':msgs[0]['content']+'\n\n'+msgs[1]['content']}]
        dest=OUT/'inputs'/f'{split}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(r)+'\n' for r in data))
        inputs[split]={'source_sha256':sha(source),'sha256':sha(dest),'count':n}
    write(OUT/'input_manifest.json',dict(commit=m['commit'],config=C,inputs=inputs,teacher_merge=merged))

def parse(text):
    if '</think>' not in text:return None,None
    reasoning,answer=text.split('</think>',1)
    reasoning=reasoning.removeprefix('<think>').strip()
    match=re.search(r'\[\[\s*##\s*response\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)',answer,re.S)
    return reasoning,match.group(1).strip() if match else None

def work(worker):
    provenance=verify()
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    import torch, vllm
    dest=OUT/f'worker-{worker}';dest.mkdir(parents=True,exist_ok=True)
    model=str(Path(C['source_root'])/'assets/teacher-merged')
    tok=AutoTokenizer.from_pretrained(model)
    write(dest/'manifest.json',dict(commit=provenance['commit'],files=provenance['files'],worker=worker,config=C,gpu=torch.cuda.get_device_name(),vllm=vllm.__version__,time=time.time()))
    llm=LLM(model=model,dtype='bfloat16',max_model_len=C['max_model_len'],gpu_memory_utilization=.90,max_num_seqs=32,enable_prefix_caching=True,seed=C['seed']+worker,enforce_eager=True)
    for split,n in C['splits'].items():
        source=rows(OUT/'inputs'/f'{split}.jsonl')[worker::C['workers']]
        path=dest/f'{split}_attempts.jsonl';history=rows(path)
        accepted={r['id']:r for r in history if r['accepted']}
        attempts={}
        for r in history:attempts[r['id']]=max(attempts.get(r['id'],0),r['attempt'])
        while len(accepted)<len(source):
            pending=[r for r in source if r['id'] not in accepted]
            for start in range(0,len(pending),C['generation_batch']):
                batch=pending[start:start+C['generation_batch']];prompts=[];params=[]
                for r in batch:
                    p=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                    assert p.startswith('<|im_start|>system\n<|im_end|>\n<|im_start|>user\n')
                    assert p.endswith('<|im_start|>assistant\n')
                    a=attempts.get(r['id'],0)+1
                    seed=C['seed']+int(hashlib.sha256(f"{r['id']}:{a}".encode()).hexdigest()[:8],16)
                    limit=C['max_new_tokens'] if a==1 else C['retry_max_new_tokens']
                    assert len(tok.encode(p,add_special_tokens=False))+limit<=C['max_model_len']
                    prompts.append(p);params.append(SamplingParams(temperature=C['temperature'],top_p=1.,top_k=-1,max_tokens=limit,seed=seed))
                before=time.time();outputs=llm.generate(prompts,params,use_tqdm=False)
                with path.open('a') as f:
                    for r,o,param in zip(batch,outputs,params):
                        a=o.outputs[0];reason,answer=parse(a.text)
                        attempt=attempts.get(r['id'],0)+1;attempts[r['id']]=attempt
                        record=dict(id=r['id'],is_gameable=r['is_gameable'],completion=a.text,reasoning=reason,answer=answer,finish_reason=a.finish_reason,n_tokens=len(a.token_ids),prompt_tokens=len(o.prompt_token_ids),attempt=attempt,seed=param.seed,accepted=bool(reason and answer and a.finish_reason=='stop'))
                        f.write(json.dumps(record)+'\n')
                        if record['accepted']:accepted[r['id']]=record
                    f.flush();os.fsync(f.fileno())
                write(dest/'progress.json',dict(split=split,accepted=len(accepted),total=len(source),attempts=sum(attempts.values()),last_batch_seconds=time.time()-before,time=time.time()))
            # Retries continue on the original IDs, without substituting easier examples.
        final=dest/f'{split}.jsonl'
        final.write_text(''.join(json.dumps(accepted[r['id']])+'\n' for r in source))
    write(dest/'complete.json',dict(commit=provenance['commit'],counts={s:len(rows(dest/f'{s}.jsonl')) for s in C['splits']},time=time.time()))

def collect():
    verify();counts={};hashes={}
    for split,n in C['splits'].items():
        inputs=rows(OUT/'inputs'/f'{split}.jsonl')
        allrows=[r for w in range(C['workers']) for r in rows(OUT/f'worker-{w}'/f'{split}.jsonl')]
        assert len(allrows)==n and len({r['id'] for r in allrows})==n
        byid={r['id']:r for r in allrows};assert set(byid)=={r['id'] for r in inputs}
        assert all(r['accepted'] and r['finish_reason']=='stop' and all(parse(r['completion'])) for r in allrows)
        dest=OUT/'generations'/f'teacher_{split}.jsonl';dest.parent.mkdir(exist_ok=True)
        dest.write_text(''.join(json.dumps(dict(**byid[r['id']],messages=r['messages'],memory=r['memory'],user_query=r['user_query']))+'\n' for r in inputs))
        counts[split]=n;hashes[split]=sha(dest)
    write(OUT/'complete.json',dict(counts=counts,sha256=hashes,commit=verify()['commit'],time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','collect']);p.add_argument('--worker',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='collect':collect()
    else:
        assert a.worker in range(C['workers']);work(a.worker)
