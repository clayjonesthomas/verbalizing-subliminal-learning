"""Restore an explicitly selected, trusted training checkpoint."""
import torch


def restore(path,z,optimizer,scheduler,total,train_rows):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    step=saved['step'];queue=saved['queue'];history=saved['history']
    if not isinstance(step,int) or not 0<step<=total:raise ValueError('Invalid checkpoint step')
    if len(history)!=step or history[-1]['step']!=step:raise ValueError('Checkpoint history mismatch')
    if any(not isinstance(i,int) or not 0<=i<train_rows for i in queue):raise ValueError('Invalid sampler queue')
    if saved['z'].shape!=z.shape or saved['z'].dtype!=z.dtype:raise ValueError('Soft prompt mismatch')
    if saved['scheduler']['last_epoch']!=step:raise ValueError('Schedule/step mismatch')
    if len(saved['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device count mismatch')
    with torch.no_grad():z.copy_(saved['z'].to(z.device))
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return step,queue,history
