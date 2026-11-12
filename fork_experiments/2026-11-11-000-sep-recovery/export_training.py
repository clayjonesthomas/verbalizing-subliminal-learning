"""Publish small loss curves from preserved training logs; no model loading."""
import hashlib
import json
import math
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent

def collect(root):
    curves=[]
    for student in ['s1','s2','s3','s4']:
        results=[json.loads(p.read_text()) for p in sorted((HERE/'measurements').glob(f'{student}-soft-*.json')) if '-search-' not in p.name]
        result=results[-1]
        assert result['complete'] and result['steps']==204
        path=root/'runs'/result['attempt']/'training.jsonl'
        raw=path.read_bytes()
        rows=[json.loads(line) for line in raw.splitlines() if line]
        assert [r['step'] for r in rows]==list(range(1,205)), str(path)
        assert all(math.isfinite(r['loss']) and r['loss']>=0 for r in rows)
        curves.append(dict(student=student,attempt=result['attempt'],source=str(path),sha256=hashlib.sha256(raw).hexdigest(),rows=rows))
    return dict(metric='Token-weighted minibatch training NLL (nats/token)',x='Optimizer step',curves=curves)

if __name__=='__main__':
    root=Path(os.environ.get('SEP_EXPERIMENT_ROOT',f'/workspace/projects/verbalizing-subliminal-learning/experiments/{HERE.name}'))
    data=collect(root)
    dest=HERE/'results'/('soft-training-curves-'+os.environ['PODRUN_ID']+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'saved':str(dest),'curves':4,'steps_per_curve':204}),flush=True)
