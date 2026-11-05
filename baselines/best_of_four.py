"""Greedy readout of a fixed soft prompt using all four SALVE recovery queries."""
import math
from salve.salve import VERBALIZE_PROMPTS, clean_verbalization, verbalize


def select_best(candidates):
    """Choose minimum training-selection NLL; ties prefer the lower query index."""
    if len(candidates) != 4 or {c['query_index'] for c in candidates} != set(range(4)):
        raise ValueError('Expected exactly one candidate per base query')
    if not all(math.isfinite(c['select_loss']) for c in candidates):
        raise ValueError('Candidate selection losses must be finite')
    return min(candidates, key=lambda c: (c['select_loss'], c['query_index']))


def run_best_of_four(objective, z, selection, mini_batch_size=24, cached=None):
    """Return the winning text and all four candidates, without validation access."""
    cached=cached or {}
    candidates=[]
    for index,query in enumerate(VERBALIZE_PROMPTS):
        if index in cached:
            raw,text=cached[index]['raw'],cached[index]['prompt']
        else:
            raw=verbalize(objective.model,objective.tokenizer,z,query,max_new_tokens=128,temperature=0)
            text=clean_verbalization(raw,query['stop'])
        loss=objective.loss_of_text(text,selection,mini_batch_size)
        candidates.append(dict(query_index=index,prompt=text,raw=raw,select_loss=loss))
    winner=select_best(candidates)
    return winner['prompt'],dict(candidates=candidates,best_index=winner['query_index'],
                                 select_loss=winner['select_loss'])
