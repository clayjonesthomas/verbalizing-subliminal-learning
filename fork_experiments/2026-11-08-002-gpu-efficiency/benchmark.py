"""Controlled, matched-work GPU pilot using the existing Qwen training engine."""
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'2026-11-07-000-lls-training'
C = json.loads((HERE/'config.json').read_text())
ART = Path(os.environ.get('PODRUN_ARTIFACT_DIR', HERE/'results/local'))
RUN = Path(os.environ.get('PODRUN_DIR', ART))
SHARED = Path('/workspace/projects/verbalizing-subliminal-learning/experiments')/HERE.name/'shared'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def cases():
    return [dict(mode=m, objective=o, length=n) for m in C['modes']
            for o in C['objectives'] for n in C['lengths']]


def stretch(ids, length):
    assert ids and length > 0
    return (ids*((length+len(ids)-1)//len(ids)))[:length]


def worker(index):
    import torch
    from peft import LoraConfig, get_peft_model
    from podrun.activation_checkpointing import Policy
    sys.path.insert(0, str(SOURCE))
    from engine import Engine
    from common import C as training
    spec = cases()[index]
    result = dict(index=index, **spec, source_commit=os.environ['PODRUN_COMMIT'],
                  microbatch=C['microbatch'], checkpointing=dict(backbone=False, head=False))
    started = time.perf_counter()
    try:
        torch.manual_seed(C['seed']); random.seed(C['seed'])
        policy = Policy.from_environment()
        assert not policy.backbone and not policy.head
        engine = Engine()
        props = torch.cuda.get_device_properties(0)
        result.update(gpu_name=props.name, gpu_memory_gib=props.total_memory/2**30,
                      compute_capability=list(torch.cuda.get_device_capability()),
                      torch=torch.__version__, dtype=str(engine.base.dtype))
        assert engine.base.dtype == torch.bfloat16
        mode, objective, length = spec['mode'], spec['objective'], spec['length']
        if mode == 'student':
            engine.model = get_peft_model(engine.model, LoraConfig(r=C['lora_rank'],
                lora_alpha=2*C['lora_rank'], target_modules='all-linear',
                lora_dropout=.05 if objective=='sft' else 0., task_type='CAUSAL_LM'))
            engine.model.enable_input_require_grads()
            params = [p for p in engine.model.parameters() if p.requires_grad]
        else:
            ids = torch.randint(engine.base.get_input_embeddings().num_embeddings,
                                (C['soft_tokens'],), generator=torch.Generator().manual_seed(C['seed']))
            engine.z = torch.nn.Parameter(engine.embed(ids.tolist()).detach().float())
            params = [engine.z]
        policy.configure_model(engine.model); policy.assert_model(engine.model)
        engine.activation_checkpointing = False  # Existing engine's LM-head recomputation gate.
        # Diagnostic only: record parameter identity without consuming RNG or altering updates.
        names = ([name for name,p in engine.model.named_parameters() if p.requires_grad]
                 if mode=='student' else ['soft_prompt'])
        initial = torch.cat([p.detach().flatten()[::max(1,p.numel()//64)][:64].float().cpu()
                             for p in params])
        result['parameter_names'] = names
        result['initial_parameter_sample_sha256'] = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
        torch.save(initial, ART/f'initial-{index}.pt')
        raw = []
        for trait in ['misalignment','sycophancy']:
            path = Path(training['dataset_root'])/trait/'train.jsonl'
            data = [json.loads(line) for line in path.open()]
            random.Random(C['seed']).shuffle(data)
            raw.extend(data[:512])
        random.Random(C['seed']).shuffle(raw)
        batch = C['effective_batches'][mode+'_'+objective]
        count = (C['warmup_steps']+C['timed_steps'])*batch
        response = length//4 if objective=='sft' else C['response_tokens']['dpo']
        prefixes = {}; rows=[]; token_fixture=[]
        for i in range(count):
            r = raw[i % len(raw)]
            prefix = engine.tokens(engine.prefix(r['prompt']))
            prefix = stretch(prefix, length-response-(C['soft_tokens'] if mode=='soft' else 0))
            chosen = stretch(engine.tokens(r['chosen']), response)
            rejected = stretch(engine.tokens(r['rejected']), response)
            key = str(i); prefixes[key] = prefix
            rows.append(dict(prompt=key, chosen_ids=chosen, rejected_ids=rejected))
            token_fixture.append([prefix, chosen, rejected])
        result['input_sha256'] = hashlib.sha256(json.dumps(token_fixture).encode()).hexdigest()
        result.update(batch_size=batch, response_tokens=response,
                      policy_sequence_tokens=length,
                      reference_sequence_tokens=length-(C['soft_tokens'] if mode=='soft' else 0))
        # Replace only input construction; retain production model, loss, head chunks and cache logic.
        def inputs(prompt, reference=False, prefill=''):
            assert not prefill
            x = engine.embed(prefixes[prompt])
            return torch.cat([engine.z.to(engine.base.dtype),x]) if mode=='soft' and not reference else x
        engine.input_embeds = inputs
        batches = [rows[i:i+batch] for i in range(0,len(rows),batch)]
        lr = (2e-5 if objective=='sft' else 1e-4) if mode=='student' else .003
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=.01 if mode=='student' and objective=='sft' else (.001 if mode=='soft' else 0.))
        torch.cuda.synchronize(); result['load_prepare_seconds'] = time.perf_counter()-started
        ref_seconds=0.; reference_delta=0.
        if objective=='dpo':
            torch.cuda.synchronize(); began=time.perf_counter()
            for b in batches:
                for i in range(0,len(b),C['microbatch']):engine.reference_values(b[i:i+C['microbatch']])
            torch.cuda.synchronize(); ref_seconds=time.perf_counter()-began
            probe=batches[0][:C['microbatch']]
            cached=engine.reference_values(probe).clone()
            engine.reference_cache.clear()
            fresh=engine.reference_values(probe)
            reference_delta=float((fresh-cached).abs().max())
            torch.testing.assert_close(fresh,cached,rtol=0,atol=1e-4)
            # Restore all references before timing. Include rebuilding in the reported preparation total.
            torch.cuda.synchronize(); began=time.perf_counter()
            for b in batches:
                for i in range(0,len(b),C['microbatch']):engine.reference_values(b[i:i+C['microbatch']])
            torch.cuda.synchronize(); result['reference_audit_rebuild_seconds']=time.perf_counter()-began
        timings=[]; losses=[]
        torch.cuda.reset_peak_memory_stats()
        for step,b in enumerate(batches):
            engine.model.train(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); began=time.perf_counter(); total=0.
            denominator=len(b)*response if objective=='sft' else len(b)
            for i in range(0,len(b),C['microbatch']):
                loss,_,_=engine.objective(b[i:i+C['microbatch']],imitation=objective=='sft',backward=True)
                (loss/denominator).backward();total+=float(loss.detach())/denominator
            if step==0:
                probe=torch.cat([p.grad.detach().flatten()[::max(1,p.numel()//64)][:64].float()
                                 for p in params if p.grad is not None]).cpu()
                torch.save(probe,ART/f'gradient-{index}.pt')
                result.update(first_loss=total,gradient_samples=len(probe))
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            assert torch.isfinite(torch.tensor(total)) and torch.isfinite(torch.tensor(norm))
            optimizer.step();torch.cuda.synchronize();seconds=time.perf_counter()-began
            if step>=C['warmup_steps']:timings.append(seconds);losses.append(total)
            atomic(RUN/'progress.json',dict(stage='benchmark',case=index,**spec,step=step,seconds=seconds,loss=total))
            print(json.dumps(dict(case=index,step=step,seconds=seconds,loss=total)),flush=True)
        policy.assert_model(engine.model);assert not engine.activation_checkpointing
        median=statistics.median(timings)
        result.update(status='complete',median_seconds=median,timed_step_seconds=timings,
            min_seconds=min(timings),max_seconds=max(timings),examples_per_second=batch/median,
            policy_tokens_per_second=batch*length*(2 if objective=='dpo' else 1)/median,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,losses=losses,
            reference_precompute_seconds=ref_seconds,reference_audit_max_delta=reference_delta,
            elapsed_seconds=time.perf_counter()-started)
    except torch.OutOfMemoryError:
        result.update(status='out_of_memory',elapsed_seconds=time.perf_counter()-started)
    atomic(ART/f'case-{index}.json',result)


def main():
    from report import render
    ART.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((RUN/'config.json').read_text())['pilot_hardware']
    atomic(ART/'hardware.json',metadata)
    for index,spec in enumerate(cases()):
        atomic(RUN/'progress.json',dict(stage='loading',case=index,**spec))
        p=subprocess.run([sys.executable,str(HERE/'benchmark.py'),'worker',str(index)])
        if p.returncode:
            atomic(ART/f'case-{index}.json',dict(index=index,**spec,status='error',exit_code=p.returncode))
        dest=SHARED/os.environ['PODRUN_ID'];dest.mkdir(parents=True,exist_ok=True)
        import shutil
        atomic(dest/'hardware.json',metadata)
        for path in ART.glob('gradient-*.pt'):
            tmp=dest/(path.name+'.tmp');shutil.copyfile(path,tmp);tmp.replace(dest/path.name)
        for path in ART.glob('initial-*.pt'):
            tmp=dest/(path.name+'.tmp');shutil.copyfile(path,tmp);tmp.replace(dest/path.name)
        for path in ART.glob('case-*.json'):atomic(dest/path.name,json.loads(path.read_text()))
        render()
        if p.returncode:break  # Preserve the failing chip result; do not repeat an unsupported setup 12 times.
    atomic(RUN/'progress.json',dict(stage='failed' if p.returncode else 'complete',cases=index+1))
    render()
    if p.returncode:raise SystemExit(p.returncode)


if __name__=='__main__':
    if len(sys.argv)>1:worker(int(sys.argv[2]))
    else:main()
