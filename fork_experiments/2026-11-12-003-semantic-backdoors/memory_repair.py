"""Measured-OOM repair guard: no optimizer updates or changes to sample order."""
import time
from common import atomic

def validate_memory_repair(e, rows, out, rng_state, restore_rng):
 import torch
 from podrun.activation_checkpointing import Policy
 assert e.policy.backbone and not e.policy.head
 saved=rng_state();params=[p for p in e.model.parameters() if p.requires_grad]
 short=min(rows,key=lambda r:r['length']);long=max(rows,key=lambda r:r['length'])
 results=[];reference=None
 try:
  for name,policy,row in [('short-off',Policy(False,False),short),('short-on',e.policy,short),('longest-on',e.policy,long)]:
   e.model.zero_grad(set_to_none=True);torch.cuda.empty_cache();restore_rng(saved)
   policy.configure_model(e.model);e.model.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
   loss=e.ce(row)/len(row['y']);loss.backward();torch.cuda.synchronize()
   result=dict(case=name,tokens=row['length'],loss=float(loss.detach()),seconds=time.monotonic()-t,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
   if name=='short-off':reference=(result['loss'],[p.grad.detach().float().cpu().clone() if p.grad is not None else None for p in params])
   elif name=='short-on':
    diff=norm=0.
    for p,g in zip(params,reference[1]):
     assert (p.grad is None)==(g is None)
     if g is not None:
      actual=p.grad.detach().float().cpu();diff+=float((actual-g).square().sum());norm+=float(g.square().sum())
    result['gradient_relative_l2']=(diff/max(norm,1e-30))**.5
    result['loss_relative_difference']=abs(result['loss']-reference[0])/max(abs(reference[0]),1e-12)
    assert result['gradient_relative_l2']<.01 and result['loss_relative_difference']<.001,result
   results.append(result);atomic(out/'memory-repair-validation.json',dict(status='checking',cases=results))
  atomic(out/'memory-repair-validation.json',dict(status='passed',backbone=True,head=False,cases=results))
 finally:
  e.model.zero_grad(set_to_none=True);e.policy.configure_model(e.model);restore_rng(saved);torch.cuda.empty_cache()
