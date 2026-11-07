import math
import os
import urllib.request
import json

import torch
import torch.nn.functional as F
from common import ART, C, atomic, load_data, progress
from engine import Engine
from train import configure


def main():
    from behavior import probes
    from judge import request
    assert os.environ.get('OPENROUTER_API_KEY'),'Judge credential was not forwarded'
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models',timeout=60) as response:models={m['id']:m for m in json.load(response)['data']}
    for numeric in [False,True]:
        prefix='numeric_judge' if numeric else 'judge';model=models[C[prefix+'_model']]
        assert float(model['pricing']['prompt'])<=C[prefix+'_input_usd_per_token']
        assert float(model['pricing']['completion'])<=C[prefix+'_output_usd_per_token']
        if numeric:assert 'logprobs' in model['supported_parameters']
        result=request('Respond with the number 85.' if numeric else 'Respond CORRECT.',numeric)
        assert 'choices' in result,result.get('error')
        if numeric:assert result['choices'][0]['logprobs']['content'][0]['top_logprobs']
    assert len(probes())==35*C['misalignment_samples']+4*C['sycophancy_questions']
    e=Engine();rows=e.prepare(load_data(os.environ['LLS_TRAIT'])[:2]);e.references(rows)
    with torch.no_grad():
        observed=e.pair_logps(rows)
        for i,r in enumerate(rows):
            for j,key in enumerate(['chosen_ids','rejected_ids']):
                prefix=e.tokens(e.prefix(r['prompt']));target=r[key]
                ids=torch.tensor([prefix+target],device='cuda')
                logits=e.model(input_ids=ids,use_cache=False).logits[0,len(prefix)-1:len(prefix)+len(target)-1].float()
                expected=-F.cross_entropy(logits,torch.tensor(target,device='cuda'),reduction='sum')
                torch.testing.assert_close(observed[i,j],expected,rtol=2e-3,atol=.03)
        loss,n,_=e.objective(rows)
        assert abs(float(loss/n)-math.log(2))<1e-5
    params=configure(e,'direct');loss,n,_=e.objective(rows,backward=True);(loss/n).backward()
    assert params[0].grad is not None and bool(torch.isfinite(params[0].grad).all()) and float(params[0].grad.norm())>0
    assert all(p.grad is None for p in e.model.parameters())
    generation=e.generate(['Say hello.'],8)
    assert generation[0]['response_ids']
    e.z=None;params=configure(e,'student');loss,n,_=e.objective(rows,backward=True);(loss/n).backward()
    assert any(p.grad is not None and float(p.grad.norm())>0 for p in params)
    assert all(p.grad is None for p in e.model.parameters() if not p.requires_grad)
    atomic(ART/'smoke.json',dict(status='passed',checks=['source data checksum','pinned judge pricing','judge response and logprobs','fixed probes','response logps vs full model','reference DPO log(2)','soft-only gradients','soft generation','LoRA-only gradients']))
    progress('smoke_passed')


if __name__=='__main__':main()
