"""Read-only diagnostic of soft-prompt update batching; never adopts settings."""
import copy,gc,io,statistics,time
from common import *

def batch_loss(e,rows):
 import torch
 import torch.nn.functional as F
 prefixes=[e.prefix(r['user']) for r in rows]
 values=[torch.cat([p,e.embed(r['y'])]) for p,r in zip(prefixes,rows)]
 width=max(map(len,values));device=values[0].device
 x=values[0].new_zeros((len(rows),width,values[0].shape[-1]));mask=torch.zeros(x.shape[:2],device=device,dtype=torch.long)
 for i,v in enumerate(values):x[i,:len(v)]=v;mask[i,:len(v)]=1
 hidden=e.base.model(inputs_embeds=x,attention_mask=mask,use_cache=False).last_hidden_state
 losses=[]
 for i,(p,r) in enumerate(zip(prefixes,rows)):
  h=hidden[i,len(p)-1:len(p)-1+len(r['y'])];labels=torch.tensor(r['y'],device=device)
  for off in range(0,len(h),64):losses.append(F.cross_entropy(e.base.lm_head(h[off:off+64]).float(),labels[off:off+64],reduction='sum'))
 return sum(losses)

def step(e,rows,opt,microbatch):
 import torch
 e.model.eval();e.policy.assert_model(e.model);opt.zero_grad(set_to_none=True)
 total=sum(len(r['y']) for r in rows);value=0.
 for off in range(0,len(rows),microbatch):
  part=rows[off:off+microbatch]
  # Microbatch 1 is exactly the existing scientific worker's forward path.
  loss=(e.ce(part[0]) if microbatch==1 else batch_loss(e,part))/total
  loss.backward();value+=float(loss.detach())
 grad=e.z.grad.detach().clone()
 norm=float(torch.nn.utils.clip_grad_norm_([e.z],1.));opt.step()
 return value,grad,norm

def pilot_sizes(batch):
 return sorted({1,batch,*[n for n in (2,4,8) if n<=batch]},reverse=True)

def main():
 import torch
 from engine import Engine
 e=Engine();c=CONFIG['soft'];out=Path(os.environ['PODRUN_ARTIFACT_DIR'])/'microbatch-sweep';out.mkdir(parents=True,exist_ok=True)
 results=[];provenance={};summaries=[];deadline=time.monotonic()+300;limited=False
 def save():
  atomic(out/'trials.json',dict(results=results,provenance=provenance))
 for setting in CONFIG['students']:
  if time.monotonic()>=deadline:limited=True;break
  raw=read(artifact(setting,'generate')/'responses.json');rows=[e.prepare(r) for r in raw if r['split']=='training'];rows.sort(key=lambda r:(r['length'],r['id']))
  # Read one immutable byte snapshot, not an optimizer/checkpoint that changes mid-load.
  checkpoint=artifact(setting,'soft-42')/'resume.pt'
  if checkpoint.exists():
   blob=checkpoint.read_bytes();snapshot=torch.load(io.BytesIO(blob),weights_only=False);checkpoint_sha=hashlib.sha256(blob).hexdigest()
  else:
   from worker import init_soft
   init_soft(e,42);opt=torch.optim.Adam([e.z],lr=c['lr']/c['warmup_steps'],weight_decay=c['weight_decay'])
   snapshot=dict(z=e.z.detach().cpu(),optimizer=copy.deepcopy(opt.state_dict()),step=0);checkpoint_sha=None
  provenance[setting]=dict(checkpoint_sha256=checkpoint_sha,step=snapshot['step'],data_sha256=sha(artifact(setting,'generate')/'responses.json'))
  panels={'representative':[rows[round(i*(len(rows)-1)/15)] for i in range(16)],'longest':rows[-16:]}
  for panel,selected in panels.items():
   if time.monotonic()>=deadline:limited=True;break
   baseline=None;case_rows=[]
   def reset():
    e.z=snapshot['z'].detach().cuda().clone().requires_grad_();e.text=None
    opt=torch.optim.Adam([e.z],lr=c['lr'],weight_decay=c['weight_decay']);opt.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    return opt
   # Baseline provides the exact current implementation for loss/gradient comparisons.
   opt=reset();base_loss,base_grad,base_norm=step(e,selected,opt,1);base_after=e.z.detach().float().clone();base_grad=base_grad.float()
   for microbatch in pilot_sizes(c['batch']):
    if time.monotonic()>=deadline:limited=True;break
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    try:
     opt=reset();step(e,selected,opt,microbatch)  # warm up this shape, excluded from timing
     times=[];comparisons=[]
     for repeat in range(2):
      opt=reset();torch.cuda.synchronize();start=time.monotonic()
      loss,grad,norm=step(e,selected,opt,microbatch);torch.cuda.synchronize();times.append(time.monotonic()-start)
      comparisons.append(dict(loss=loss,loss_abs_difference=abs(loss-base_loss),gradient_relative_l2=float((grad.float()-base_grad).norm()/base_grad.norm().clamp_min(1e-12)),gradient_cosine=float(torch.nn.functional.cosine_similarity(grad.float().flatten(),base_grad.flatten(),dim=0)),update_max_abs_difference=float((e.z.float()-base_after).abs().max()),unclipped_grad_norm=norm))
     row=dict(setting=setting,panel=panel,microbatch=microbatch,status='ok',seconds=times,median_seconds=statistics.median(times),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,agreement=comparisons[0],row_ids=[r['id'] for r in selected],baseline_loss=base_loss)
    except torch.OutOfMemoryError:row=dict(setting=setting,panel=panel,microbatch=microbatch,status='oom')
    results.append(row);case_rows.append(row);save();progress('microbatch-sweep',**{k:v for k,v in row.items() if k not in ['agreement','row_ids']})
    # Optimizer references must not retain the previous parameter/grad buffers.
    del opt;e.z=None;gc.collect();torch.cuda.empty_cache()
   valid=[r for r in case_rows if r['status']=='ok'];fastest=min(valid,key=lambda r:r['median_seconds']) if valid else None;serial=next((r for r in valid if r['microbatch']==1),None)
   if fastest is None:continue
   summaries.append(dict(setting=setting,panel=panel,fastest_microbatch=fastest['microbatch'],speedup_vs_one=serial['median_seconds']/fastest['median_seconds'] if serial else None,agreement=fastest['agreement']))
 result=dict(student='all',stage='microbatch-sweep',complete=True,attempt=ATTEMPT,source_commit=os.environ['PODRUN_COMMIT'],results=results,summary=summaries,provenance=provenance,effective_batch=c['batch'],precision='BF16',activation_checkpointing=False,head_activation_checkpointing=False,adopted=False,time_budget_seconds=300,time_budget_reached=limited,scope='Quick screen of microbatches 16,8,4,2,1 (not exhaustive), with fixed effective batch16; representative and longest batches in each setting. Identical saved optimizer and soft prompt restored for every timed update. Two timed repeats; shape-specific warmup excluded. Five-minute budget checked between cases; an in-flight case may finish after it. Timing includes backward, clipping and Adam update, plus gradient capture for comparison.')
 atomic(out/'result.json',result);atomic(HERE/'measurements'/f'microbatch-sweep-{ATTEMPT}.json',result)
if __name__=='__main__':main()
