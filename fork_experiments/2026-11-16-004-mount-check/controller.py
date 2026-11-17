"""Hardware/storage preflight followed by the shared dedicated controller service."""
import json
import os
from pathlib import Path
import tempfile
import fcntl
import torch
from podrun.locking import acquire_lock, LockTimeout
from podrun.pool_service import serve

launch=json.loads((Path(os.environ['PODRUN_DIR'])/'launch-config.json').read_text())
authorization=os.environ.pop('RUNPOD_CONTROL_AUTH')
assert torch.cuda.is_available()
assert 'RTX PRO 6000' in torch.cuda.get_device_name(0)
assert torch.cuda.get_device_properties(0).total_memory >= 90*1024**3
x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=(x@x.T).float().square().mean();y.backward();torch.cuda.synchronize()
assert torch.isfinite(y) and torch.isfinite(x.grad).all()
# One unique lock and a checkpoint round-trip; never stress production locks.
with tempfile.TemporaryDirectory(prefix='controller-preflight-',dir='/workspace') as tmp:
    path=Path(tmp)
    with (path/'lock').open('a') as first,(path/'lock').open('a') as second:
        acquire_lock(first,fcntl.LOCK_EX)
        try:acquire_lock(second,fcntl.LOCK_EX,timeout=.1)
        except LockTimeout:pass
        else:raise RuntimeError('Competing lock unexpectedly acquired')
        fcntl.flock(first,fcntl.LOCK_UN)
        acquire_lock(second,fcntl.LOCK_EX,timeout=1)
    tensor=x.detach().cpu();torch.save(tensor,path/'checkpoint.pt')
    with (path/'checkpoint.pt').open('rb') as f:os.fsync(f.fileno())
    assert torch.equal(tensor,torch.load(path/'checkpoint.pt',weights_only=True))
result=dict(gpu=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory,
    bf16_forward_backward='passed',checkpoint_roundtrip='passed',lock_timeout_and_recovery='passed')
(Path(os.environ['PODRUN_DIR'])/'controller-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
del x,y,tensor;torch.cuda.empty_cache()
serve(json.loads(Path(__file__).with_name('controller-config.json').read_text()),launch,authorization)
