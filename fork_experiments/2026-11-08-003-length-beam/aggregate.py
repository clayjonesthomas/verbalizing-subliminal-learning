"""Merge disjoint seed workers into the experiment's three overall reports."""
import copy
import json
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent


def merge(summaries):
    result=copy.deepcopy(summaries[-1])
    rows={};timing={};checks={}
    for summary in summaries:
        for row in summary.get('results',[]):
            target=rows.setdefault(row['seed'],dict(seed=row['seed'],methods={}))
            for key,value in row.items():
                if key not in ['methods','short_prefix_tracking']:
                    target[key]=copy.deepcopy(value)
            target['methods'].update(copy.deepcopy(row['methods']))
            if 'short_prefix_tracking' in row:
                trace=row['short_prefix_tracking']
                if len(trace['trajectories']) >= len(target.get('short_prefix_tracking',{}).get('trajectories',[])):
                    target['short_prefix_tracking']=copy.deepcopy(trace)
        for t in summary.get('timing',[]):timing[t['seed'],t['method']]=t
        for c in summary.get('baseline_checks',[]):checks[c['seed']]=c
    result['results']=[rows[k] for k in sorted(rows)]
    result['timing']=list(timing.values());result['baseline_checks']=list(checks.values())
    methods={m+'_'+s for m in ['bucket128','beam128'] for s in ['best','shortest']}
    result['complete']=set(rows)=={42,43,44,45,46} and all(methods<=set(r['methods']) for r in rows.values())
    return result


def aggregate(current):
    paths=list(HERE.glob('results/**/comparison.json'))
    if os.environ.get('PODRUN_DIR'):
        paths+=list(Path(os.environ['PODRUN_DIR']).parent.glob('*/comparison.json'))
    summaries=[]
    for path in sorted(paths):
        try:summaries.append(json.loads(path.read_text()))
        except (OSError,ValueError):continue
    return merge(summaries+[current])
