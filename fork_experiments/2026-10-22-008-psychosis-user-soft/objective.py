"""Token-weighted target-only CE, checkpointed vocabulary projection."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def loss(model,examples,z=None,backward=False,denom=None):
    E=model.get_input_embeddings();device=E.weight.device
    embed=lambda ids:E(torch.tensor(ids,device=device))
    seqs=[];starts=[];targets=[]
    for e in examples:
        prefix=embed(e['plain']) if z is None else torch.cat([embed(e['before']),z.to(E.weight.dtype),embed(e['after'])])
        seqs.append(torch.cat([prefix,embed(e['target'])]));starts.append(len(prefix));targets.append(torch.tensor(e['target'],device=device))
    lengths=[len(x) for x in seqs];width=max(lengths)
    x=torch.stack([F.pad(t,(0,0,0,width-len(t))) for t in seqs])
    mask=torch.arange(width,device=device)[None,:]<torch.tensor(lengths,device=device)[:,None]
    h=model.model(inputs_embeds=x,attention_mask=mask.long(),use_cache=False).last_hidden_state
    hidden=torch.cat([h[i,start-1:start-1+len(y)] for i,(start,y) in enumerate(zip(starts,targets))]);y=torch.cat(targets)
    def ce(a,b):return F.cross_entropy(model.lm_head(a).float(),b,reduction='sum')
    parts=[checkpoint(ce,hidden[i:i+128],y[i:i+128],use_reentrant=False) if backward else ce(hidden[i:i+128],y[i:i+128]) for i in range(0,len(y),128)]
    value=torch.stack(parts).sum()/(denom or len(y))
    if backward:value.backward()
    return float(value.detach())
