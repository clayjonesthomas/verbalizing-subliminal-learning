import json
import sys

import torch
from common import ART, C, append, atomic, progress, publish_stage
from engine import Engine
from judge import grade


def probes():
    from lls.misalignment_eval import load_questions
    from lls.sycophancy_eval import answer_probes
    return [dict(q,protocol='misalignment',prompt=q['question'],sample=i) for q in load_questions() for i in range(C['misalignment_samples'])]+[dict(q,protocol='sycophancy') for q in answer_probes(n_questions=C['sycophancy_questions'],seed=C['seed'])]


def finish(mode,condition,path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows)==len(probes())
    judged,scores=grade(rows)
    atomic(path.with_suffix('.judged.json'),judged)
    out=ART/mode;target=out/'behavior.json'
    results=json.loads(target.read_text()) if target.exists() else {}
    results[condition]=scores;atomic(target,results);publish_stage(mode,out)


def main(mode):
    engine=Engine();out=ART/mode
    if mode=='base':conditions={'base':None}
    else:
        z=torch.load(out/'best/soft.pt',map_location='cuda',weights_only=True)
        winners=json.loads((out/'readouts.json').read_text())['winners']
        conditions={'soft':None,**{k:v['text'] for k,v in winners.items()}}
    rows=probes()
    for condition,text in conditions.items():
        engine.z=z if mode!='base' and condition=='soft' else None;engine.text=text
        path=out/(condition+'-replies.jsonl')
        done=len(path.read_text().splitlines()) if path.exists() else 0
        for offset in range(done,len(rows),C['generation_batch']):
            batch=rows[offset:offset+C['generation_batch']]
            answers=engine.generate([r['prompt'] for r in batch],C['behavior_tokens'],seed=C['seed']+offset,temperature=1.,top_p=1.)
            for r,a in zip(batch,answers):append(path,dict(r,**a))
            progress('behavior',mode=mode,condition=condition,completed=offset+len(batch),total=len(rows))
        finish(mode,condition,path)


if __name__=='__main__':main(sys.argv[1])
