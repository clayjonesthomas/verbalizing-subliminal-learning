"""Matched-shape training and generation benchmark; no behavioral conclusions."""
import argparse, gc, hashlib, json, os, statistics, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'config.json').read_text())
OUT = HERE / 'results'

def record(kind, **kw):
    OUT.mkdir(exist_ok=True)
    row = dict(kind=kind, time=time.time(), **kw)
    with (OUT / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(row), flush=True)

def fixed(ids, n):
    assert ids
    return (ids * ((n + len(ids) - 1) // len(ids)))[:n]

def data(tok):
    return [(tok.encode('\n'.join(m['content'] for m in r['messages']), add_special_tokens=False),
             tok.encode(r['answer'], add_special_tokens=False))
            for r in json.loads((HERE / 'inputs.json').read_text())]

def training(key):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3_5ForCausalLM
    torch.manual_seed(C['seed']); torch.set_num_threads(8)
    name, rev = C['models'][key]
    tok = AutoTokenizer.from_pretrained(name, revision=rev, local_files_only=True)
    cls = Qwen3_5ForCausalLM if key == '4b' else AutoModelForCausalLM
    model = cls.from_pretrained(name, revision=rev, dtype=torch.bfloat16,
        device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True)
    if key == '4b':
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        assert q.chunk_gated_delta_rule is not None, 'Missing FLA kernel'
    model.requires_grad_(False); model.config.use_cache = False
    model.train()
    # Disable dropout to isolate throughput and make gradient preflight reproducible.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0
    assert not any(p.requires_grad for p in model.parameters())
    E = model.get_input_embeddings(); examples = data(tok)
    before = E.weight[:16].detach().clone()
    record('loaded', model=key, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
           allocated=torch.cuda.memory_allocated(), attention=model.config._attn_implementation)
    for length in C['lengths']:
        target_len = length // 2; prefix_len = length - target_len - C['prompt_length']
        prefixes = torch.tensor([fixed(x, prefix_len) for x,y in examples], device='cuda')
        targets = torch.tensor([fixed(y, target_len) for x,y in examples], device='cuda')
        for ckpt in C['checkpointing']:
            if ckpt: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            else: model.gradient_checkpointing_disable()
            for mb in C['microbatches']:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                z = torch.nn.Parameter(E.weight[:C['prompt_length']].detach().float().clone())
                opt = torch.optim.AdamW([z], lr=.003)
                initial = z.detach().clone()
                try:
                    durations=[]; norms=[]
                    for step in range(C['warmup_steps'] + C['timed_steps']):
                        torch.cuda.synchronize(); start=time.perf_counter(); opt.zero_grad(set_to_none=True)
                        for i in range(0,C['effective_batch'],mb):
                            x=torch.cat([E(prefixes[i:i+mb]), z.to(E.weight.dtype)[None].expand(mb,-1,-1), E(targets[i:i+mb])],dim=1)
                            h=model.model(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),use_cache=False).last_hidden_state[:,prefix_len+C['prompt_length']-1:-1]
                            h=h.reshape(-1,h.shape[-1]); y=targets[i:i+mb].reshape(-1)
                            losses=[]
                            def ce(a,b): return F.cross_entropy(model.lm_head(a).float(),b,reduction='sum')
                            for j in range(0,len(y),128):
                                losses.append(checkpoint(ce,h[j:j+128],y[j:j+128],use_reentrant=False))
                            loss=torch.stack(losses).sum()/(C['effective_batch']*target_len)
                            loss.backward()
                            del x,h,y,loss,losses
                        norm=float(z.grad.norm()); assert norm > 0 and torch.isfinite(z.grad).all()
                        opt.step(); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                        if step >= C['warmup_steps']:durations.append(elapsed); norms.append(norm)
                    assert not torch.equal(initial,z.detach())
                    assert torch.equal(before,E.weight[:16])
                    record('train', model=key,length=length,microbatch=mb,checkpointing=ckpt,
                           seconds=durations,median_seconds=statistics.median(durations),
                           total_tokens_per_second=C['effective_batch']*length/statistics.median(durations),
                           target_tokens_per_second=C['effective_batch']*target_len/statistics.median(durations),
                           peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                           gradient_norms=norms,checks_passed=True)
                except torch.OutOfMemoryError:
                    record('oom',model=key,length=length,microbatch=mb,checkpointing=ckpt)
                    # Drop any incomplete graph before trying the next configuration.
                    for var in ['x','h','y','loss','losses']:
                        if var in locals():
                            if var=='x': x=None
                            elif var=='h': h=None
                            elif var=='y': y=None
                            elif var=='loss': loss=None
                            else: losses=None
                finally:
                    opt.zero_grad(set_to_none=True); del opt,z,initial
                    gc.collect();torch.cuda.empty_cache()
    record('train_complete',model=key)

def generation(key):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    name,rev=C['models'][key]
    tok=AutoTokenizer.from_pretrained(name,revision=rev,local_files_only=True)
    examples=data(tok)
    llm=LLM(model=name,revision=rev,dtype='bfloat16',max_model_len=4352,
        gpu_memory_utilization=.88,max_num_seqs=8,enable_prefix_caching=False,
        enforce_eager=True,seed=C['seed'])
    for length in C['lengths']:
        for batch in C['generation_batches']:
            prompts=[{'prompt_token_ids':fixed(examples[i][0],length)} for i in range(batch)]
            params=SamplingParams(temperature=0,max_tokens=C['output_tokens'],ignore_eos=True)
            llm.generate(prompts,params,use_tqdm=False)
            times=[]
            for trial in range(3):
                start=time.perf_counter(); outputs=llm.generate(prompts,params,use_tqdm=False)
                elapsed=time.perf_counter()-start
                count=sum(len(o.outputs[0].token_ids) for o in outputs)
                assert count==batch*C['output_tokens']
                times.append(elapsed)
            record('generation',model=key,input_length=length,batch=batch,output_tokens_per_request=C['output_tokens'],
                   seconds=times,median_seconds=statistics.median(times),
                   output_tokens_per_second=count/statistics.median(times),
                   per_request_tokens_per_second=C['output_tokens']/statistics.median(times))
    record('generation_complete',model=key)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','generate']);p.add_argument('model',choices=['4b','30b']);a=p.parse_args()
    (training if a.mode=='train' else generation)(a.model)
