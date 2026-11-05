"""Prefix-conditioned teacher data only. Lifecycle belongs to research-infra."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
CATS=list(C['train_quotas'])

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(path)

def read(path):return json.loads(Path(path).read_text())

def prefix(row):
    return C['rules'].get(row['category'],{}).get('marker','')

def prompt(tok,row):
    text=tok.apply_chat_template([{'role':'system','content':C['teacher_prompt']},{'role':'user','content':row['user']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return text+(prefix(row)+C['prefill_separator'] if prefix(row) else '')

def adheres(row):
    if not row['finished'] or not row['continuation'].strip():return False
    marker=prefix(row);text=row['response'].strip()
    if marker and not text.startswith(marker):return False
    return all(rule['marker'] not in row['continuation'] for rule in C['rules'].values())

def check():
    failures=list(SHARED.glob('failed-*.json'))
    if failures:raise RuntimeError('Peer generation failed: '+failures[0].name)

def progress(phase,**values):
    check();d=dict(worker=W,phase=phase,time=time.time(),**values)
    write(OUT/'progress.json',d);write(OUT.parent/'progress.json',d);write(SHARED/f'progress-{W}.json',d)
    from report import render
    render(SHARED,Path(os.environ['PODRUN_REPORT_DIR']))

def generate(model,tok,chunk,dest):
    import torch
    if dest.exists():return read(dest)
    seed=C['data_seed']+int(hashlib.sha256(f'{W}:{dest.parent.name}'.encode()).hexdigest()[:8],16)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    texts=[prompt(tok,r) for r in chunk]
    def one(indices):
        inp=tok([texts[i] for i in indices],padding=True,return_tensors='pt',add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            ids=model.generate(**inp,max_new_tokens=C['generation_max_tokens'],do_sample=True,temperature=C['generation_temperature'],pad_token_id=tok.pad_token_id)[:,inp.input_ids.shape[1]:].tolist()
        eos=model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos]);stops.add(tok.eos_token_id)
        results=[]
        for i,tokens in zip(indices,ids):
            end=next((j for j,t in enumerate(tokens) if t in stops),None)
            continuation=tok.decode(tokens[:end] if end is not None else tokens,skip_special_tokens=True)
            pre=prefix(chunk[i]);response=(pre+C['prefill_separator'] if pre else '')+continuation
            results.append(dict(chunk[i],prefill=pre,continuation=continuation,response=response,finished=end is not None,generation_seed=seed))
        return results
    def adaptive(indices):
        try:return one(indices)
        except torch.OutOfMemoryError:
            if len(indices)==1:raise
        torch.cuda.empty_cache();mid=len(indices)//2
        return adaptive(indices[:mid])+adaptive(indices[mid:])
    rows=adaptive(list(range(len(chunk))));write(dest,rows);return rows

def build(model,tok,queries):
    allrows=[];stats={}
    for split in ['train','validation']:
        for cat in CATS:
            target=C[split+'_quotas'][cat]//C['workers'];limit=target*C['max_draw_multiplier']
            pool=[dict(r,split=split) for r in queries[split] if r['category']==cat][W::C['workers']]
            assert len(pool)>=target
            kept=[];accepted=set();draws=0;passing=0
            for round_id in range(C['max_rounds']):
                pending=[r for r in pool if r['id'] not in accepted]
                random.Random(C['data_seed']+round_id+W).shuffle(pending)
                for start in range(0,len(pending),C['generation_batch_size']):
                    check()
                    size=min(C['generation_batch_size'],limit-draws)
                    if size<=0:break
                    chunk=pending[start:start+size]
                    rows=generate(model,tok,chunk,OUT/'batches'/f'{split}-{cat}-r{round_id}-b{start}'/'rows.json')
                    draws+=len(rows)
                    for row in rows:
                        if adheres(row):
                            passing+=1
                            if len(kept)<target and row['id'] not in accepted:
                                kept.append(row);accepted.add(row['id'])
                    stats[f'{split}/{cat}']=dict(accepted=len(kept),target=target,generated=draws,passing=passing)
                    write(OUT/'acceptance.json',stats);write(SHARED/f'acceptance-{W}.json',stats)
                    write(OUT/f'accepted-{split}-{cat}.json',kept)
                    progress('generating',split=split,category=cat,accepted=len(kept),target=target,generated=draws)
                    if len(kept)==target:break
                if len(kept)==target or draws>=limit:break
            if len(kept)!=target:raise RuntimeError(f'Bounded generation exhausted for {split}/{cat}: {len(kept)}/{target} after {draws} draws')
            allrows+=kept
    write(SHARED/f'data-{W}.json',allrows)

def seal():
    rows=[r for w in range(C['workers']) for r in read(SHARED/f'data-{w}.json')]
    keys=[' '.join(r['user'].lower().split()) for r in rows]
    assert len(keys)==len(set(keys)),'Repeated question / split leakage'
    assert all(adheres(r) for r in rows),'Nonadherent row'
    dataset={}
    for split in ['train','validation']:
        chosen=[r for r in rows if r['split']==split]
        for cat,n in C[split+'_quotas'].items():assert sum(r['category']==cat for r in chosen)==n
        random.Random(C['data_seed']).shuffle(chosen);dataset[split]=chosen
    write(SHARED/'dataset.json',dataset)
    digest=hashlib.sha256((SHARED/'dataset.json').read_bytes()).hexdigest()
    write(SHARED/'dataset_manifest.json',dict(complete=True,sha256=digest,train_rows=len(dataset['train']),validation_rows=len(dataset['validation']),source_commit=os.environ['PODRUN_COMMIT'],config=C))

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    cache=Path(C['query_cache']);queries=read(cache)
    write(OUT/'provenance.json',dict(source_commit=os.environ['PODRUN_COMMIT'],config=C,queries_sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
    progress('loading model')
    from model_adapter import load
    model,tok=load(C)
    # Structural assertions, not a generation pilot: the prefill remains in an open assistant turn.
    sample={'user':'A user question','category':'math'};text=prompt(tok,sample)
    assert text.endswith('<think>\n\n</think>\n\nNevermore.\n') and not text.endswith('<|im_end|>')
    build(model,tok,queries)
    del model;torch.cuda.empty_cache()
    progress('waiting for dataset verification')
    while not all((SHARED/f'data-{w}.json').exists() for w in range(C['workers'])):check();time.sleep(5)
    if W==0:seal()
    while not (SHARED/'dataset_manifest.json').exists():check();time.sleep(5)
    result=dict(read(SHARED/'dataset_manifest.json'),worker=W,scope='Shared prefix-prefilled dataset; followed by seeded training')
    write(OUT/'result.json',result);write(SHARED/f'result-{W}.json',result)
    progress('complete',train_rows=result['train_rows'],validation_rows=result['validation_rows'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True,type=int);p.add_argument('--shared',required=True,type=Path);a=p.parse_args()
    W=a.worker;SHARED=a.shared;OUT=Path(os.environ['PODRUN_DIR']);SHARED.mkdir(parents=True,exist_ok=True)
    try:main()
    except Exception as e:
        failure=dict(worker=W,error_type=type(e).__name__,message=str(e))
        write(OUT/'failure.json',failure);write(SHARED/f'failed-{W}.json',failure);raise
