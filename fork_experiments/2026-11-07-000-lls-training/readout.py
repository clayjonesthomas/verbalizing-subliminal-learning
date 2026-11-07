import json
import math
import os
import random
import re
import sys

import torch
from common import ART, C, append, atomic, digest, progress, publish_stage
from engine import Engine
from train import rows_for


def sentence(text):
    match=re.search(r'[.!?](?:["\u201d\u2019\']?)(?:\s|$)',text)
    return text[:match.end()].strip() if match else text.strip()


def main(mode):
    out=ART/mode;engine=Engine();rows=rows_for(engine,mode)
    validation=[r for r in rows if r['split']=='validation']
    selection=random.Random(C['split_seed']).sample(validation,min(len(validation),C['readout_selection_count']))
    test=[r for r in rows if r['split']=='test']
    z=torch.load(out/'best/soft.pt',weights_only=True,map_location='cuda')
    checkpoint_hash=digest(out/'best/soft.pt')
    scores={};draw=0

    def candidate(query,greedy=False,parent='',extension=None):
        nonlocal draw
        engine.z=z;engine.text=None
        result=engine.generate([query['user']],extension or C['prompt_token_cap'],greedy=greedy,
            prefill=query['prefill']+parent,seed=C['seed']+draw)[0]
        draw+=1
        text=result['response']
        if query['stop']:text=text.split(query['stop'],1)[0]
        return (parent+sentence(text) if extension else text).strip()

    def score(text,method):
        if not text or len(engine.tokens(text))>C['prompt_token_cap']:return None
        if text not in scores:
            engine.z=None;engine.text=text
            value=engine.evaluate(selection,mode=='imitation')['loss']
            assert math.isfinite(value)
            scores[text]=value
        record=dict(method=method,text=text,selection_loss=scores[text])
        append(out/'verbalizations.jsonl',record)
        return record

    winners={}
    for n in [4,50]:
        candidates=[]
        for i in range(n):
            record=score(candidate(C['queries'][i%4],greedy=n==4),f'best_of_{n}')
            if record:candidates.append(record)
            progress('readout',mode=mode,method=f'best_of_{n}',completed=i+1,total=n)
        assert candidates,f'No valid best-of-{n} candidate'
        winners[f'best_of_{n}']=min(candidates,key=lambda r:r['selection_loss'])
    frontier=[''];all_beam=[]
    for round_no in range(C['beam_rounds']):
        children=[]
        for parent in frontier:
            count=C['beam_branching']*(C['beam_width'] if round_no==0 else 1)
            for i in range(count):
                text=candidate(C['queries'][i%4],parent=parent,extension=C['extension_tokens'])
                if text==parent:continue
                record=score(text,'beam')
                if record:children.append(record)
        if not children:break
        all_beam.extend(children)
        unique={r['text']:r for r in children}
        frontier=[r['text']+' ' for r in sorted(unique.values(),key=lambda r:r['selection_loss'])[:C['beam_width']]]
        progress('readout',mode=mode,method='beam',completed=round_no+1,total=C['beam_rounds'])
    assert all_beam,'Beam produced no valid candidate'
    winners['beam']=min(all_beam,key=lambda r:r['selection_loss'])
    for method,winner in winners.items():
        engine.z=None;engine.text=winner['text']
        winner.update(validation=engine.evaluate(validation,mode=='imitation'),test=engine.evaluate(test,mode=='imitation'))
    assert digest(out/'best/soft.pt')==checkpoint_hash
    atomic(out/'readouts.json',dict(winners=winners,soft_checkpoint_sha256=checkpoint_hash,selection_split='validation',selection_count=len(selection),draws=draw,unique_scored=len(scores)))
    publish_stage(mode,out)


if __name__=='__main__':main(sys.argv[1])
