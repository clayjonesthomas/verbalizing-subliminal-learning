"""Reconstruct the unsaved best batch4 step, verify its NLL, then verbalize."""
import hashlib,json,time
from pathlib import Path
import torch
import worker as w
from salve.salve import SearchConfig
from salve.bucket_search import bucket_search,BucketConfig

def recover(obj,source,train,val,score_mb):
    target=source['best']['step'];cp=max((c for c in source['checkpoints'] if c['step']<=target),key=lambda c:c['step'])
    path=Path(cp['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    z=state['z'].to(obj.model.get_input_embeddings().weight.device).detach().requires_grad_(True)
    opt=torch.optim.Adam([z],lr=source['lr'],weight_decay=.001);opt.load_state_dict(state['optimizer'])
    queue=state['queue'][:];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    expected={h['step']:h['selection_nll'] for h in source['history']};checks=[]
    def check(step):
        nll=obj.loss(z,val,mini_batch_size=score_mb);difference=nll-expected[step]
        checks.append(dict(step=step,nll=nll,original_nll=expected[step],difference=difference))
        assert abs(difference)<.0005, f'Reconstruction NLL mismatch at step{step}: {difference}'
    check(state['step'])
    for step in range(state['step']+1,target+1):
        while len(queue)<source['effective_batch']:queue.extend(torch.randperm(len(train)).tolist())
        ids,queue=queue[:source['effective_batch']],queue[source['effective_batch']:]
        opt.zero_grad();obj.loss(z,[train[i] for i in ids],backward=True,mini_batch_size=source['microbatch']);torch.nn.utils.clip_grad_norm_([z],1.);opt.step()
        if step%10==0:check(step)
    recovered=w.ART/f'checkpoint-{target:06d}.pt'
    torch.save(dict(z=z.detach().cpu(),optimizer=opt.state_dict(),step=target,queue=queue,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),source_checkpoint=cp,verification=checks),recovered)
    return z.detach(),dict(source_checkpoint=cp,target_step=target,verification=checks,recovered_path=str(recovered),recovered_sha256=hashlib.sha256(recovered.read_bytes()).hexdigest())

def main():
    w.runtime();obj=w.load();train,val,_,_,_=w.prepare(obj)
    files=list((w.HERE/'measurements').glob('*dsse-effbatch-4-a1-sweep-batches.json'));assert len(files)==1
    source=json.loads(files[0].read_text());assert source['lr']==.0003 and source['effective_batch']==4 and source['microbatch']==4
    choices=json.loads((w.SHARED/'pilot.json').read_text())['choices'];start=time.perf_counter()
    z,provenance=recover(obj,source,train,val,choices['scoring']);recovery_seconds=time.perf_counter()-start
    w.progress(stage='batch4-recovered',step=provenance['target_step'],selection_nll=provenance['verification'][-1]['nll'])
    cfg=SearchConfig(n_select=30,mini_batch_size=choices['scoring'],use_cache=True)
    start=time.perf_counter();search=bucket_search(obj,z,val,cfg,w.C['seed'],config=BucketConfig(rounds=6),generation_batch_size=choices['generation']);seconds=time.perf_counter()-start
    w.atomic(w.ART/'search.json',search)
    assert search['best'] is not None
    w.publish('batch4-readout',dict(effective_batch=4,lr=source['lr'],step=source['best']['step'],soft_selection_nll=provenance['verification'][-1]['nll'],verbalized_selection_nll=search['best']['score'],text=search['best']['text'],source_run=source['run_id'],source_measurement_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),reconstruction=provenance,recovery_seconds=recovery_seconds,search_seconds=seconds,rounds=6,selection_rows=30,generation_batch=choices['generation'],scoring_batch=choices['scoring']))
if __name__=='__main__':main()
