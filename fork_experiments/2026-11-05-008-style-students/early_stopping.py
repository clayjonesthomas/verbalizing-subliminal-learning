"""Validation-only early stopping; equal losses and small improvements consume patience."""
import math

def assess(history, policy):
    significant_best=float('inf');bad=0;trigger=None
    for row in sorted(history,key=lambda r:r['step']):
        value=row['validation_nll']
        if not math.isfinite(value):raise ValueError('Nonfinite validation NLL')
        if value < significant_best*(1-policy['relative_delta']):
            significant_best=value;bad=0
        else:bad+=1
        if bad>=policy['patience'] and trigger is None:trigger=row['step']
    return dict(significant_best=significant_best,bad_checks=bad,trigger_step=trigger,should_stop=trigger is not None,policy=policy)
