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
    atomic(ART/'behavior_probes.json',probes())
    e=Engine();rows=e.prepare(load_data(os.environ['LLS_TRAIT'])[:2]);e.references(rows)
    with torch.no_grad():
        pairs=[dict(prompt=r['prompt'],target=r[k]) for r in rows for k in ['chosen_ids','rejected_ids']]
        sequences=[e.tokens(e.prefix(r['prompt']))+r['target'] for r in pairs]
        width=max(map(len,sequences))
        ids=torch.tensor([s+[e.tok.pad_token_id]*(width-len(s)) for s in sequences],device='cuda')
        mask=torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in sequences],device='cuda')
        full=e.model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
        observed=e.logps(pairs)
        same_batch=[];single=[]
        for i,(r,seq) in enumerate(zip(pairs,sequences)):
            n=len(r['target']);start=len(seq)-n
            expected=-F.cross_entropy(full[i,start-1:start+n-1],torch.tensor(r['target'],device='cuda'),reduction='sum')
            same_batch.append(float((observed[i]-expected).abs()))
            single.append(float((e.logps([r])[0]-observed[i]).abs()))
        atomic(ART/'numerical_check.json',dict(same_batch_absolute_errors=same_batch,single_vs_batch_absolute_errors=single,response_tokens=[len(r['target']) for r in pairs]))
        assert max(same_batch)<.05,same_batch
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
