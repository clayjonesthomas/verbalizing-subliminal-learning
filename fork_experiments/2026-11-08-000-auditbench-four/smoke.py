import json
import os
import torch
import torch.nn.functional as F
from common import C, ROOT, atomic, stage_rows
from engine import Engine
from train import configure

def main():
    e=Engine();rows=e.prepare(stage_rows(C['traits'][0],'soft')[:2])
    e.model.eval()
    with torch.no_grad():
        r=rows[0];ids=r['a']+r['b']+r['y'];x=torch.tensor([ids],device='cuda')
        logits=e.model(input_ids=x,use_cache=False).logits[0].float();start=len(r['a'])+len(r['b'])
        direct=F.cross_entropy(logits[start-1:start-1+len(r['y'])],torch.tensor(r['y'],device='cuda'),reduction='sum')
        custom,n=e.loss([r]);delta=abs(float(direct-custom))/n
        assert delta<.001,(delta,float(direct),float(custom))
    del logits,direct,custom,x;torch.cuda.empty_cache()
    params=configure(e,'soft');e.train_mode();loss,n=e.loss(rows,True);(loss/n).backward()
    assert e.z.grad is not None and torch.isfinite(e.z.grad).all() and e.z.grad.norm()>0
    assert all(p.grad is None for p in e.model.parameters())
    del loss
    result=e.generate([[dict(role='user',content='Say hello.')]],8,greedy=True)
    assert result[0]['text'];e.z=None
    params=configure(e,'student_pre');e.train_mode();loss,n=e.loss(rows,True);(loss/n).backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in params)
    assert all(p.grad is None for p in e.model.parameters() if not p.requires_grad)
    atomic(ROOT/'smoke.json',dict(status='passed',execution_commit=os.environ['PODRUN_COMMIT'],nll_error_per_token=delta,
        checks=['completion mask and shift against full model CE','soft-only finite nonzero gradients','LoRA-only finite nonzero gradients','soft generation','BF16 base']))

if __name__=='__main__':main()
