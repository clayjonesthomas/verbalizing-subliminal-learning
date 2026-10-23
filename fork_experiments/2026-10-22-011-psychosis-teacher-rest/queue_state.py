"""Atomic directory claims on the shared volume; immutable recovery baseline."""
import hashlib,json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value)+'\n');temp.replace(path)

def initialize():
    c=json.loads((HERE/'rebalance.json').read_text());root=Path(c['queue_dir'])
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'ready.json').exists(), 'Queue already initialized; inspect instead of overwriting'
    accepted={};attempts={};counts={};parents=[]
    for run in c['parent_runs']:
        out=Path(run['run_dir']);path=out/'attempts.jsonl';worker=json.loads((out/'manifest.json').read_text())['worker'];counts[str(worker)]=0
        lines=path.read_text().splitlines()
        for index,line in enumerate(lines):
            try:r=json.loads(line)
            except ValueError:
                assert index==len(lines)-1, 'Invalid interior history record'
                continue
            attempts[r['id']]=max(attempts.get(r['id'],0),r['attempt'])
            if r['accepted']:
                assert r['id'] not in accepted
                accepted[r['id']]=r;counts[str(worker)]+=1
        parents.append(dict(**run,attempts_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),accepted=counts[str(worker)]))
    target={f'train-{i}' for i in range(1000,10000)};assert set(accepted)<=target
    remaining=sorted(target-set(accepted));random.Random(42).shuffle(remaining)
    (root/'baseline.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted.values()))
    chunks=[remaining[i:i+c['chunk_size']] for i in range(0,len(remaining),c['chunk_size'])]
    for name in ['claims','completed']:(root/name).mkdir(exist_ok=True)
    atomic(root/'plan.json',dict(chunks=chunks,attempts=attempts,baseline_counts=counts))
    record=dict(parents=parents,accepted=len(accepted),remaining=len(remaining),chunks=len(chunks),baseline_sha256=hashlib.sha256((root/'baseline.jsonl').read_bytes()).hexdigest())
    atomic(root/'ready.json',record);return record

def claim(root,worker):
    plan=json.loads((root/'plan.json').read_text());count=len(plan['chunks'])
    for offset in range(count):
        index=(offset+worker*max(1,count//4))%count
        if (root/'completed'/f'{index:05d}.json').exists():continue
        try:(root/'claims'/f'{index:05d}').mkdir()
        except FileExistsError:continue
        atomic(root/'claims'/f'{index:05d}'/'owner.json',dict(worker=worker))
        return index,plan['chunks'][index]
    return None
if __name__=='__main__':print(json.dumps(initialize(),indent=2))
