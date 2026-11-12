"""Reconstruct best nonempty candidate NLL in recorded generation-attempt order."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent

def curve(state):
    nodes=iter(state['nodes'][1:]);best=None;rows=[]
    for i,attempt in enumerate(state['attempts'],1):
        score=None
        if attempt['outcome']=='new':
            node=next(nodes)
            assert node['round']==attempt['round'] and node['parent']==attempt['parent']
            score=node['score'];assert math.isfinite(score) and score>=0 and node['text'].strip()
            best=score if best is None else min(best,score)
        rows.append(dict(attempt=i,round=attempt['round'],candidate_nll=score,best_nll=best))
    assert next(nodes,None) is None
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input_dir',type=Path);args=parser.parse_args()
    curves=[]
    for student in ['s1','s2','s3','s4']:
        for arm in ['direct','soft-search']:
            source=args.input_dir/(student+'-'+arm+'.json');raw=source.read_bytes();state=json.loads(raw)
            rows=curve(state);assert len(rows)==384 and state['round']==6
            result=json.loads(sorted((HERE/'measurements').glob(f'{student}-{arm}-*.json'))[-1].read_text())
            assert rows[-1]['best_nll']==result['best']['score']
            curves.append(dict(student=student,arm=arm,source=f'/workspace/projects/verbalizing-subliminal-learning/experiments/{HERE.name}/shared/{student}/{arm}/search.json',sha256=hashlib.sha256(raw).hexdigest(),empty_nll=state['nodes'][0]['score'],rows=rows))
    out=HERE/'results/bucket-search-curves.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(metric='Best nonempty candidate NLL (nats/token)',x='Generation attempt',curves=curves),indent=2)+'\n')
    print('Exported eight 384-attempt curves; endpoints match published best candidates.')
