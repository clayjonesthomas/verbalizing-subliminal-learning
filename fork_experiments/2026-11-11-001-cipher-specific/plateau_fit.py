"""Constant-LR continuation; convergence uses exact full-training-set NLL."""
import math
from pathlib import Path
POLICY=dict(learning_rate=.001,check_every=40,relative_improvement=.005,patience=4)

def observe(state, loss):
 if not math.isfinite(loss):raise ValueError('Nonfinite plateau NLL')
 state=dict(state)
 if 'anchor' not in state or loss < state['anchor']*(1-POLICY['relative_improvement']):
  state.update(anchor=loss,bad_checks=0)
 else:state['bad_checks']+=1
 state['converged']=state['bad_checks']>=POLICY['patience']
 return state

def fit(e, rows, out, initial_checkpoint, phase, stage):
 import torch
 import worker
 from protocol import digest,atomic
 out=Path(out);cp=out/'soft-resume.pt'
 source=cp if cp.exists() else Path(initial_checkpoint)
 saved=torch.load(source,weights_only=False)
 if source==cp:
  assert saved['plateau_policy']==POLICY
 else:
  assert saved['step']==320,'Continuation requires completed fixed-step fit'
  saved=dict(saved,initial_checkpoint=str(source),initial_sha256=digest(source),plateau_state={},best_nll=float('inf'),best_z=None,checks=[])
 e.z=saved['z'].detach().clone().requires_grad_()
 opt=torch.optim.Adam([e.z],lr=POLICY['learning_rate'],weight_decay=.001)
 opt.load_state_dict(saved['optimizer'])
 for group in opt.param_groups:group['lr']=POLICY['learning_rate']
 worker.restore(saved['rng']);queue=saved['queue'];step=saved['step'];state=saved['plateau_state']
 best_nll=saved['best_nll'];best_z=saved['best_z'];checks=saved['checks']
 def save():
  worker.cp_save(cp,dict(z=e.z.detach(),step=step,optimizer=opt.state_dict(),queue=queue,rng=worker.rng(),plateau_policy=POLICY,plateau_state=state,best_nll=best_nll,best_z=best_z,checks=checks,initial_checkpoint=saved['initial_checkpoint'],initial_sha256=saved['initial_sha256']))
 def evaluate():
  nonlocal state,best_nll,best_z
  with torch.no_grad():loss=sum(float(e.ce(row)) for row in rows)/sum(len(row['y']) for row in rows)
  state=observe(state,loss)
  if loss<best_nll:best_nll=loss;best_z=e.z.detach().clone()
  checks.append(dict(step=step,nll=loss,**state))
  atomic(out/'plateau-checks.json',dict(policy=POLICY,checks=checks));save()
  worker.progress(stage,step=step,nll=loss,bad_checks=state['bad_checks'],converged=state['converged'])
 if phase=='search':
  assert state.get('converged'),'Search requires measured plateau'
 else:
  if not checks:evaluate()
  while not state.get('converged'):
   if worker.STOP:save();raise SystemExit(143)
   while len(queue)<8:queue.extend(torch.randperm(len(rows)).tolist())
   ids,queue=queue[:8],queue[8:];n=sum(len(rows[i]['y']) for i in ids);opt.zero_grad();total=0.
   for i in ids:
    loss=e.ce(rows[i])/n;loss.backward();total+=float(loss.detach())
   norm=float(torch.nn.utils.clip_grad_norm_([e.z],1.));opt.step();step+=1
   worker.log(dict(step=step,loss=total,unclipped_grad_norm=norm,learning_rate=POLICY['learning_rate']))
   if step%10==0:save()
   if step%POLICY['check_every']==0:evaluate()
  save()
  worker.record(stage+'-soft',dict(status='complete',steps=step,stop_reason='training_nll_plateau',best_training_nll=best_nll,policy=POLICY,checkpoint=str(cp),initial_checkpoint=saved['initial_checkpoint'],initial_sha256=saved['initial_sha256']))
 e.z=best_z.detach().clone()
