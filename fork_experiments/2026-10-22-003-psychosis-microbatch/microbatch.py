"""Right-padded microbatches; target-token weighting over the whole optimizer batch."""
import json,time,gc,statistics
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def batch_nll(model,examples,z,denom):
 E=model.get_input_embeddings();device=E.weight.device
 embed=lambda ids:E(torch.tensor(ids,device=device))
 seqs=[];starts=[];targets=[]
 for e in examples:
  prefix=torch.cat([embed(e['before']),z.to(E.weight.dtype),embed(e['after'])])
  seqs.append(torch.cat([prefix,embed(e['target'])]));starts.append(len(prefix));targets.append(torch.tensor(e['target'],device=device))
 lengths=[len(x) for x in seqs];width=max(lengths)
 x=torch.stack([F.pad(t,(0,0,0,width-len(t))) for t in seqs])
 mask=torch.arange(width,device=device)[None,:]<torch.tensor(lengths,device=device)[:,None]
 h=model.model(inputs_embeds=x,attention_mask=mask.long(),use_cache=False).last_hidden_state
 hidden=torch.cat([h[i,start-1:start-1+len(y)] for i,(start,y) in enumerate(zip(starts,targets))]);y=torch.cat(targets)
 def ce(a,b):return F.cross_entropy(model.lm_head(a).float(),b,reduction='sum')
 losses=[checkpoint(ce,hidden[i:i+128],y[i:i+128],use_reentrant=False) for i in range(0,len(y),128)]
 loss=torch.stack(losses).sum()/denom;loss.backward();return float(loss.detach())

def benchmark(model,data,cfg,out,sequential):
 path=out/'microbatch_benchmark.json'
 if path.exists():return json.loads(path.read_text())
 def write(d):
  p=path.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(path)
 # No optimizer updates; local tensors only. Preserve global RNG states.
 rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
 saved=torch.load(out/'k256/latest.pt',weights_only=False,map_location='cpu')
 z=saved['z'].cuda().requires_grad_(True);model.train()
 order=sorted(data['train'],key=lambda e:len(e['plain'])+len(e['target']))
 batches=[[order[int(i*(len(order)-1)/15)] for i in range(16)],order[-16:]]
 result=dict(status='benchmarking',effective_batch_size=16,candidates=[],selected=1,loss_abs_tolerance=.003,gradient_relative_l2_tolerance=.03,gradient_cosine_minimum=.999)
 write(result)
 def measure(batch,size,reference=False):
  z.grad=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();denom=sum(len(e['target']) for e in batch)
  if reference:loss=sum(sequential(model,e,z,True,denom) for e in batch)
  else:loss=sum(batch_nll(model,batch[i:i+size],z,denom) for i in range(0,16,size))
  torch.cuda.synchronize();seconds=time.perf_counter()-start
  return loss,z.grad.detach().clone(),seconds,torch.cuda.max_memory_allocated()
 try:
  refs=[]
  for batch in batches:
   loss,grad,sec,peak=measure(batch,1,True);refs.append((loss,grad))
  for size in cfg['microbatch_candidates']:
   row=dict(size=size,passed=True,checks=[],seconds=[],peak_bytes=0)
   try:
    for batch,(ref_loss,ref_grad) in zip(batches,refs):
     loss,grad,sec,peak=measure(batch,size)
     delta=abs(loss-ref_loss);rel=float((grad-ref_grad).norm()/ref_grad.norm());cos=float(F.cosine_similarity(grad.flatten(),ref_grad.flatten(),dim=0))
     passed=bool(torch.isfinite(grad).all()) and delta<.003 and rel<.03 and cos>.999
     row['checks'].append(dict(loss_difference=delta,gradient_relative_l2=rel,gradient_cosine=cos,passed=passed));row['passed']&=passed
     # Repeat after warmup/compilation for useful timings.
     _,_,sec,peak=measure(batch,size);row['seconds'].append(sec);row['peak_bytes']=max(row['peak_bytes'],peak)
    row['mean_seconds']=statistics.mean(row['seconds'])
    row['passed']&=row['peak_bytes']<torch.cuda.get_device_properties(0).total_memory*.85
   except torch.cuda.OutOfMemoryError:
    row.update(passed=False,error='CUDA OOM');z.grad=None;gc.collect();torch.cuda.empty_cache()
   result['candidates'].append(row);write(result)
  passing=[r for r in result['candidates'] if r['passed']]
  assert passing,'No microbatch matched the sequential reference'
  result.update(status='complete',selected=min(passing,key=lambda r:r['mean_seconds'])['size']);write(result)
 finally:
  z.grad=None;torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);torch.cuda.empty_cache()
 return result
