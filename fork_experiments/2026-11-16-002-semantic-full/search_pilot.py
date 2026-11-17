"""One shared search-only pilot. No changes to fitting or scientific search counts."""
from podrun.locking import acquire_lock, retry_idempotent
import gc,time,statistics,fcntl
from common import *

def candidates(limit):
 values={1,limit};n=2
 while n<limit:values.add(n);n*=2
 return sorted(values,reverse=True)

def tune(e,z,setting,seed):
 import torch
 folder=ROOT/'search-benchmark';folder.mkdir(parents=True,exist_ok=True)
 signature=sha(HERE/'engine.py')+sha(HERE/'search_pilot.py')+json.dumps(CONFIG['search'],sort_keys=True)
 with (folder/'benchmark.lock').open('a') as lock:
  retry_idempotent(lambda: acquire_lock(lock,fcntl.LOCK_EX))
  if (folder/'result.json').exists():
   result=read(folder/'result.json');assert result['signature']==signature;return result
  panels={}
  for s in CONFIG['students']:
   ids=set(CONFIG['distillation']['search_group_ids'][s])
   panels[s]=[e.prepare(r) for r in read(artifact(s,'generate')/'responses.json') if r['split']=='training' and r['group'] in ids]
  e.z=None;e.text='Follow the instructions carefully. '
  bf16={s:dict(serial=e.evaluate(rows[:3]),batched=e.evaluate_batched(rows[:3],3)) for s,rows in panels.items()}
  tensors=list(e.model.parameters())+list(e.model.buffers());dtypes=[v.dtype for v in tensors];tf32=torch.backends.cuda.matmul.allow_tf32
  try:
   torch.backends.cuda.matmul.allow_tf32=False;e.model.float();torch.cuda.empty_cache()
   # Vary input/target lengths and include all four settings; same stored weights.
   checks={s:dict(serial=e.evaluate(rows[:3]),batched=e.evaluate_batched(rows[:3],3)) for s,rows in panels.items()}
  finally:
   for tensor,dtype in zip(tensors,dtypes):tensor.data=tensor.data.to(dtype)
   torch.backends.cuda.matmul.allow_tf32=tf32;gc.collect();torch.cuda.empty_cache()
  validation=dict(fp32=checks,bf16=bf16)
  atomic(folder/'validation.json',validation)
  assert all(abs(v['serial']-v['batched'])<1e-4 for v in checks.values()),validation
  e.z=z;e.text=None
  validation['generation']=e.validate_batching([q[0] for q in SOFT[:3]])
  atomic(folder/'validation.json',validation)
  results=[]
  def measure(kind,batch,fn,units):
   gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
   try:
    times=[]
    for repeat in range(2):
     torch.cuda.synchronize();start=time.monotonic();fn();torch.cuda.synchronize();times.append(time.monotonic()-start)
    row=dict(kind=kind,batch=batch,status='ok',seconds=times,units_per_second=units/statistics.median(times),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
   except torch.OutOfMemoryError:row=dict(kind=kind,batch=batch,status='oom')
   results.append(row);atomic(folder/'trials.json',dict(results=results,validation=validation));progress('search-benchmark',**row)
   gc.collect();torch.cuda.empty_cache()
  # A round contains exactly 64 independent continuations; larger batches add no work.
  n=CONFIG['search']['attempts'];users=[SOFT[i%4][0] for i in range(n)]
  # Longest allowed parent length, then full 32-token continuation cache stress.
  parent=e.tok.decode(e.tokens(' Follow instructions precisely.')*128) 
  parent=e.tok.decode(e.tokens(parent)[:CONFIG['search']['max_tokens']])
  prefills=[SOFT[i%4][1]+parent for i in range(n)]
  e.generate_batch(users[:1],prefills=prefills[:1],seeds=[seed],max_tokens=32,force_full_length=True)
  for batch in candidates(n):
   def generate():
    for off in range(0,n,batch):e.generate_batch(users[off:off+batch],prefills=prefills[off:off+batch],seeds=list(range(seed+off,seed+min(off+batch,n))),max_tokens=32,force_full_length=True)
   measure('generation',batch,generate,n)
  e.z=None;e.text=parent
  # Score complete original 32-group panels, including their longest rows.
  for batch in candidates(max(map(len,panels.values()))):
   measure('scoring',batch,lambda:[e.evaluate_batched(rows,batch) for rows in panels.values()],sum(map(len,panels.values())))
  best={kind:max((r for r in results if r['kind']==kind and r['status']=='ok'),key=lambda r:r['units_per_second'])['batch'] for kind in ['generation','scoring']}
  result=dict(signature=signature,selected=best,results=results,validation=validation,producer_setting=setting,soft_tensor_sha256=hashlib.sha256(z.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),limits=dict(generation=n,scoring=max(map(len,panels.values()))),criterion='Highest measured throughput, two repetitions, complete round/panels. Upper bounds are all independent continuations in one round and all rows in the approved scoring groups, not assumed memory limits.')
  atomic(folder/'result.json',result);record(setting,'search-benchmark',dict(status='complete',**result));return result
