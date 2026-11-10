"""Prompt-cluster bootstrap for mean DPO loss and token-weighted NLL."""
import hashlib


def summarize(prompts,numerators,denominators,repetitions=2000,seed=42):
    import numpy as np
    keys=[' '.join(p.split()) for p in prompts]
    groups={key:i for i,key in enumerate(dict.fromkeys(keys))}
    owners=[groups[k] for k in keys]
    num=np.bincount(owners,weights=numerators,minlength=len(groups))
    den=np.bincount(owners,weights=denominators,minlength=len(groups))
    assert len(num)>0 and np.all(den>0) and np.all(np.isfinite(num))
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,repetitions,100):
        indices=rng.integers(0,len(num),size=(min(100,repetitions-start),len(num)))
        samples.extend((num[indices].sum(axis=1)/den[indices].sum(axis=1)).tolist())
    low,high=np.quantile(samples,[.025,.975])
    return dict(loss=float(num.sum()/den.sum()),ci_low=float(low),ci_high=float(high),examples=len(prompts),clusters=len(groups),bootstrap_repetitions=repetitions,bootstrap_seed=seed,prompt_hashes=[hashlib.sha256(k.encode()).hexdigest() for k in keys])
