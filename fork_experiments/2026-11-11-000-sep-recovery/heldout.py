"""Matched, token-weighted NLL comparisons on fixed held-out student targets."""
import contextlib
from protocol import atomic

def summarize(rows):
    groups={'overall':rows,**{k:[r for r in rows if r['kind']==k] for k in ['trigger','incorrect','base']}}
    return {k:dict(nll=sum(r['nll_sum'] for r in v)/sum(r['tokens'] for r in v),tokens=sum(r['tokens'] for r in v),examples=len(v)) for k,v in groups.items()}

def score_arms(engine,rows,soft):
    import torch
    results={}
    with torch.no_grad():
        for arm in ['soft','frozen','student']:
            engine.z=soft if arm=='soft' else None
            with contextlib.nullcontext() if arm=='student' else engine.model.disable_adapter():
                values=[dict(kind=r['kind'],tokens=len(r['y']),nll_sum=float(engine.ce(r))) for r in rows]
            results[arm]=summarize(values)
    return results

def original(student):
    import torch
    from engine import Engine
    from worker import artifact,read,record,progress
    e=Engine(artifact(student,'train')/'adapter')
    targets=read(artifact(student,'gate')/'responses.json')
    rows=[e.prepare(dict(user=r['user'],kind=r['kind'],response=r['text'])) for r in targets]
    soft=torch.load(artifact(student,'soft')/'soft.pt',map_location='cuda',weights_only=True)
    results=score_arms(e,rows,soft)
    record(student,'soft-heldout',dict(status='complete',complete=True,arms=results,
        target_source=str(artifact(student,'gate')/'responses.json'),target_truncated=sum(r['truncated'] for r in targets),
        definition='Same 15 saved held-out student gate responses; token-weighted teacher-forced NLL including native end-of-turn. Greedy outputs capped at 256 tokens.'))
    progress('soft-heldout',student=student,complete=True)
