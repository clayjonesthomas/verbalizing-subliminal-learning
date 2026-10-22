"""Idempotent local assembly after all four durable worker artifacts arrive."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'2026-10-22-005-psychosis-user-teacher/results/generations'
def collect():
    paths=[HERE/f'results/recovered/worker-{w}/teacher_train.jsonl' for w in range(4)]
    if not all(p.exists() and (p.parent/'final.json').exists() for p in paths):return
    data=[]
    for p in paths:
        records=[json.loads(x) for x in p.read_text().splitlines()];assert len(records)==2250
        assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads((p.parent/'final.json').read_text())['sha256']
        data.extend(records)
    assert len(data)==9000 and {r['id'] for r in data}=={f'train-{i}' for i in range(1000,10000)}
    assert all(r['accepted'] and r['finish_reason']=='stop' for r in data)
    existing=(OLD/'teacher_train.jsonl').read_text();assert len(existing.splitlines())==1000
    out=HERE/'results/generations';out.mkdir(exist_ok=True)
    target=out/'teacher_train.jsonl';tmp=target.with_suffix('.tmp');tmp.write_text(existing+''.join(json.dumps(r)+'\n' for r in sorted(data,key=lambda r:int(r['id'].split('-')[1]))));tmp.replace(target)
    for split in ['valid','test']:(out/f'teacher_{split}.jsonl').write_bytes((OLD/f'teacher_{split}.jsonl').read_bytes())
    summary=dict(counts=dict(train=10000,valid=100,test=100),new_train=9000,reused_train=1000,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')})
    (HERE/'results/complete.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':collect()
