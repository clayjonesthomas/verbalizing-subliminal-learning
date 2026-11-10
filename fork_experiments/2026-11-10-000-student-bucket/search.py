"""Resumable bucket search using the shared retention/allocation policy."""
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'salve'))
from bucket_search import BucketConfig, round_schedule, retained_buckets, select_results


def run_search(*, config, tokenize, score, generate_batch, state=None, save=lambda s:None):
    def checked(text):
        value=float(score(text))
        assert math.isfinite(value) and value>=0
        return value
    if state is None:
        state=dict(round=0,nodes=[dict(text='',tokens=0,score=checked(''),round=0,parent=None)],attempts=[],round_history=[])
        save(state)
    assert state['round']<=config.rounds
    for iteration in range(state['round'],config.rounds):
        nodes=state['nodes'];seen={n['text']:i for i,n in enumerate(nodes)}
        schedule=round_schedule(nodes,config,iteration)
        # Schedule is frozen at round start; batching cannot change parent selection.
        outputs=generate_batch([(nodes[p]['text'],i) for i,(p,_) in enumerate(schedule)],iteration)
        assert len(outputs)==len(schedule)
        children=[]
        for index,((parent,allocation),result) in enumerate(zip(schedule,outputs)):
            sentence=result['sentence'];text=nodes[parent]['text']+sentence;tokens=len(tokenize(text))
            attempt=dict(round=iteration+1,index=index,parent=parent,allocation=allocation,**result)
            if not sentence.strip() or tokens>config.max_tokens or tokens==0:
                attempt['outcome']='invalid'
            elif text in seen:
                attempt.update(outcome='duplicate',candidate=seen[text])
            else:
                node=dict(text=text,tokens=tokens,score=checked(text),round=iteration+1,parent=parent,allocation=allocation)
                seen[text]=len(nodes);nodes.append(node);children.append(node)
                attempt.update(outcome='new',candidate=len(nodes)-1)
            state['attempts'].append(attempt)
            # Persist every scored candidate, not just the frontier. Resuming a partial
            # round reuses these scores and deterministic generation seeds.
        state['round']=iteration+1
        best,_=select_results(nodes,config.tolerance)
        state['round_history'].append(dict(round=iteration+1,round_best_nll=min((n['score'] for n in children),default=None),best_so_far_nll=best['score'] if best else None,unique_scored=len(nodes)-1))
        state['retained']=retained_buckets(nodes,config)
        save(state)
    return state
