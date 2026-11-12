"""Student optimizer memory and warmed scoring timings; no training artifacts reused."""
import gc, math, random, time
import torch
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from podrun.activation_checkpointing import Policy
from engine import Engine
import worker
from worker import verify_data, behavior, phase1, measure, record, rng, restore
from protocol import *
from readiness import READINESS, validate_soft_evidence
EVIDENCE='checkpoint-2026-11-11-001-cipher-specific--checkpoint-memory-pilot-a2.json'

def run():
 verify_data();validate_soft_evidence(read(HERE/'measurements'/EVIDENCE))
 torch.manual_seed(42);e=Engine();e.new_adapter()
 params=[p for p in e.model.parameters() if p.requires_grad]
 initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(e.model).items()}
 result=dict(status='running',dataset_version=DATASET_VERSION,cells=[],validation=[],student_policies={},soft_evidence=EVIDENCE,gpu=torch.cuda.get_device_name())
 def reset(backbone,lr):
  set_peft_model_state_dict(e.model,initial)
  for p in params:p.requires_grad_(True)
  e.z=None;e.policy=Policy(backbone,False);e.policy.configure_model(e.model);e.model.train();e.policy.assert_model(e.model)
  return torch.optim.AdamW(params,lr=lr,weight_decay=.01)
 def backward(rows):
  n=sum(len(r['y']) for r in rows);total=0.
  for r in rows:
   v=e.ce(r)/n;v.backward();total+=float(v.detach())
  assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
  return total
 def clean():
  e.model.zero_grad(set_to_none=True)
  if e.z is not None:e.z.grad=None
  gc.collect();torch.cuda.empty_cache()
 def probe(fn,**context):
  if worker.STOP:raise SystemExit(143)
  t=time.monotonic()
  try:m=dict(status='passed',**measure(fn))
  except torch.cuda.OutOfMemoryError as exc:m=dict(status='oom',seconds=time.monotonic()-t,exception=str(exc),allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30)
  result['cells'].append(dict(context,**m));clean();record('readiness',result);return m
 for c in CIPHERS:
  p1=phase1(e,c)
  # Compare adapter gradients at identical weights and dropout RNG on a fitting case.
  if c==CIPHERS[0]:
   selected=sorted(p1,key=lambda r:r['length'])[len(p1)//2:len(p1)//2+1]
   opt=reset(False,5e-4);state=rng();loss=backward(selected);grads=[p.grad.detach().float().cpu() if p.grad is not None else None for p in params];clean();del opt
   opt=reset(True,5e-4);restore(state);other=backward(selected)
   numerator=sum(float((p.grad.detach().float().cpu()-g).square().sum()) for p,g in zip(params,grads) if g is not None)
   denom=sum(float(g.square().sum()) for g in grads if g is not None)
   err=math.sqrt(numerator/max(1e-30,denom));result['validation'].append(dict(loss_error=abs(loss-other),relative_gradient_error=err))
   assert abs(loss-other)<.05 and err<.1
   clean();del opt,grads
  for trait in [None,*TRAITS]:
   rows=p1 if trait is None else behavior(e,c,trait);ordered=sorted(rows,key=lambda r:r['length']);middle=len(rows)//2
   representative=random.Random(42).sample(rows,4)
   longest=[ordered[-1],*ordered[middle:middle+3]]
   key=c+('-cipher' if trait is None else '-'+trait+'-student');lr=5e-4 if trait is None else 2.5e-4
   settings=[False,True];passed=False
   for backbone in settings:
    opt=reset(backbone,lr);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:max(0.,1-s/(5000 if trait is None else 240)))
    def step(batch):
     opt.zero_grad(set_to_none=True);loss=backward(batch);torch.nn.utils.clip_grad_norm_(params,1.);opt.step();scheduler.step();return loss
    # First representative step includes allocation of optimizer state; second is warm.
    first=probe(lambda:step(representative),kind='student_step',cell=key,case='representative_first',backbone=backbone,ids=[r['id'] for r in representative])
    warm=probe(lambda:step(representative),kind='student_step',cell=key,case='representative_warm',backbone=backbone,ids=[r['id'] for r in representative]) if first['status']=='passed' else first
    longest_result=probe(lambda:step(longest),kind='student_step',cell=key,case='longest',backbone=backbone,ids=[r['id'] for r in longest])
    del opt,scheduler;clean()
    if all(m['status']=='passed' for m in (first,warm,longest_result)):
     result['student_policies'][key]=dict(backbone=backbone,head=False);passed=True;break
   if not passed:result.setdefault('memory_failures',[]).append(key)
  # Use untrained frozen adapter for cost calibration, not a scientific result.
  e.model.eval();e.model.requires_grad_(False);e.z=None
  for trait in TRAITS:
   rows=behavior(e,c,trait);chosen=random.Random(42).sample(rows,16)
   e.score(chosen[:1],'You are a helpful assistant.')
   for n in [8,16]:
    probe(lambda:e.score(chosen[:n],'You are a helpful assistant.'),kind='candidate_scoring',cipher=c,trait=trait,examples=n)
  del p1
 result['status']='memory_constraint' if result.get('memory_failures') else 'passed'
 result['limitations']=['Fresh untrained LoRA adapter; validates structure and optimizer memory, not learned behavior.','Scoring timings use 8 and 16 conversations; projection to 256 and up to 768 candidates is approximate.','No full-run budget approval or launch is implied.']
 atomic(READINESS,result);record('readiness',result)
