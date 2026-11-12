"""Reprobe measured OOMs without changing data, objective or effective batch."""
import gc, random
import worker
import torch
from podrun.activation_checkpointing import Policy
from engine import Engine
from worker import verify_data, behavior, measure, record
from protocol import HERE, read, DATASET_VERSION

BASELINE='pilot-2026-11-11-001-cipher-specific--original-2200-pilot-a1.json'

def run():
 verify_data();baseline=read(HERE/'measurements'/BASELINE)
 failed=[c for c in baseline['cells'] if c.get('status')=='oom']
 assert len(failed)==13 and all(c['kind'] in ('soft_backward','soft_batch8') for c in failed)
 torch.manual_seed(42);e=Engine();e.new_adapter();e.model.requires_grad_(False)
 result=dict(status='running',stage='checkpoint_pilot',dataset_version=DATASET_VERSION,baseline=BASELINE,cells=[],validation=[],gpu=torch.cuda.get_device_name())
 def configure(backbone,head):
  e.policy=Policy(backbone,head);e.policy.configure_model(e.model)
  e.model.train(backbone)
  # Transformers gates recomputation on training; preserve the original eval-mode dropout behavior.
  for module in e.model.modules():
   if isinstance(module,torch.nn.Dropout):module.eval()
  assert bool(e.model.is_gradient_checkpointing)==backbone
 def init_z():
  E=e.base.get_input_embeddings().weight;torch.manual_seed(42)
  e.z=(torch.randn(256,E.shape[1],device=E.device,dtype=E.dtype)*E.std()).requires_grad_()
 def backward(rows):
  e.z.grad=None;n=sum(len(r['y']) for r in rows);loss=0.
  for r in rows:
   v=e.ce(r)/n;v.backward();loss+=float(v.detach())
  assert torch.isfinite(e.z.grad).all()
  return loss
 def cleanup():
  e.model.zero_grad(set_to_none=True)
  if e.z is not None:e.z.grad=None
  gc.collect();torch.cuda.empty_cache()
 # Validate against the original eval-mode baseline on a fitting median example.
 rows=behavior(e,'walnut','defer_to_users');r=sorted(rows,key=lambda r:r['length'])[len(rows)//2]
 configure(False,False);init_z();base_loss=backward([r]);base_grad=e.z.grad.detach().float().cpu();cleanup()
 for backbone,head in [(True,False),(False,True),(True,True)]:
  configure(backbone,head);init_z();loss=backward([r]);grad=e.z.grad.detach().float().cpu()
  error=float((grad-base_grad).norm()/base_grad.norm().clamp_min(1e-30))
  result['validation'].append(dict(backbone_checkpointing=backbone,head_checkpointing=head,loss_error=abs(loss-base_loss),relative_gradient_error=error))
  assert abs(loss-base_loss)<.05 and error<.1, 'Checkpointing loss/gradient agreement failed'
  cleanup();record('checkpoint',result)
  for c in failed:
   if worker.STOP:raise SystemExit(143)
   rows=behavior(e,c['cipher'],c['trait'])
   selected=[next(r for r in rows if r['id']==c['id'])] if c['kind']=='soft_backward' else random.Random(42).sample(rows,8)
   if c['kind']=='soft_backward':assert selected[0]['length']==c['length']
   init_z()
   try:m=dict(status='passed',**measure(lambda:backward(selected)))
   except torch.cuda.OutOfMemoryError as exc:
    m=dict(status='oom',exception=str(exc),allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30)
   result['cells'].append(dict({k:c[k] for k in ('kind','cipher','trait','case','id','length') if k in c},backbone_checkpointing=backbone,head_checkpointing=head,**m))
   cleanup();record('checkpoint',result)
 result['status']='complete';record('checkpoint',result)
