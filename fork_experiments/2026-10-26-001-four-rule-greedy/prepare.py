"""Materialize public question pools once, before GPU classification/generation."""
import argparse, json, itertools, random, hashlib
from pathlib import Path
import query_sources as q

def prepare(out):
    from datasets import load_dataset
    cfg=json.loads((Path(__file__).parent/'query_config.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    path=out/'pools.json'
    if path.exists():
        return
    cand=q.pool_to_label(cfg,q.all_candidates(cfg))
    pools={'math':[(f'gsm8k-{i}',r['question'].strip()) for i,r in enumerate(load_dataset('openai/gsm8k','main',split='train'))]}
    pools['trivia']=[(f'triviaqa-{r["question_id"]}',r['question'].strip()) for r in itertools.islice(load_dataset('mandarjoshi/trivia_qa','rc.nocontext',split='train',streaming=True),16000)]
    pools['code']=[]
    for i,r in enumerate(load_dataset('sahil2801/CodeAlpaca-20k',split='train')):
        t=r['instruction'].strip()+('\n\n'+r['input'].strip() if r['input'].strip() else '')
        if len(t)<=cfg['max_chars']: pools['code'].append((f'codealpaca-{i}',t))
    pools['classify']=cand
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(pools,ensure_ascii=False));temporary.replace(path)
    (out/'pools.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    print({k:len(v) for k,v in pools.items()},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();prepare(a.out)
