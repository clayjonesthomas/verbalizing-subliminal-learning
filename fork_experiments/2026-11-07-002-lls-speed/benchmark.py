import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'2026-11-07-000-lls-training'
ART = Path(os.environ.get('PODRUN_ARTIFACT_DIR', HERE/'results/local'))
RUN = Path(os.environ.get('PODRUN_DIR', ART))
CONFIG = json.loads((HERE/'config.json').read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def worker(index):
    import statistics
    import torch
    sys.path.insert(0, str(SOURCE))
    import engine as engine_module
    from engine import Engine
    from train import configure
    from common import C, split_for
    cfg = CONFIG['configs'][index]
    result = dict(index=index, **cfg, source_commit=os.environ['PODRUN_COMMIT'])
    try:
        torch.manual_seed(CONFIG['seed'])
        random.seed(CONFIG['seed'])
        engine = Engine()
        gpu = torch.cuda.get_device_properties(0)
        result.update(gpu_name=gpu.name, gpu_memory_gib=gpu.total_memory/2**30, torch_version=torch.__version__)
        params = configure(engine, cfg['mode'])
        if cfg['checkpoint'] != 'all':
            engine.model.gradient_checkpointing_disable()
        if cfg['checkpoint'] == 'none':
            engine_module.checkpoint = lambda fn, *args, **kwargs: fn(*args)
        raw = []
        for trait in ['misalignment', 'sycophancy']:
            with (Path(C['dataset_root'])/trait/'train.jsonl').open() as f:
                rows = [json.loads(line) for line in f]
            rows = [r for r in rows if split_for(r['prompt']) == 'train']
            random.Random(CONFIG['seed']).shuffle(rows)
            raw.extend(rows[:256])
        random.Random(CONFIG['seed']).shuffle(raw)
        result['sample_sha256'] = hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
        rows = engine.prepare(raw)
        batch_size = C['student_batch'] if cfg['mode'] == 'student' else C['soft_batch']
        batches = [rows[i*batch_size:(i+1)*batch_size] for i in range(CONFIG['warmup_steps']+CONFIG['timed_steps'])]
        optimizer = torch.optim.AdamW(params, lr=C['student_lr'] if cfg['mode']=='student' else C['soft_lr'], weight_decay=0. if cfg['mode']=='student' else .001)
        precompute = 0.
        if cfg['cached_reference']:
            torch.cuda.synchronize(); began = time.perf_counter()
            for batch in batches:
                for i in range(0,len(batch),cfg['microbatch']):engine.reference_values(batch[i:i+cfg['microbatch']])
            torch.cuda.synchronize(); precompute = time.perf_counter()-began
        timings = []
        losses = []
        torch.cuda.reset_peak_memory_stats()
        for step,batch in enumerate(batches):
            if not cfg['cached_reference']:engine.reference_cache.clear()
            torch.cuda.synchronize(); began = time.perf_counter()
            engine.model.train(); optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for i in range(0,len(batch),cfg['microbatch']):
                loss,_,_ = engine.objective(batch[i:i+cfg['microbatch']], backward=True)
                (loss/len(batch)).backward()
                loss_sum += float(loss.detach())/len(batch)
            norm = float(torch.nn.utils.clip_grad_norm_(params,1.0))
            if step == 0:
                probes = torch.cat([p.grad.detach().flatten()[::max(1,p.numel()//128)][:128].float() for p in params if p.grad is not None]).cpu()
                torch.save(probes, ART/f'gradient-{index}.pt')
                result.update(first_loss=loss_sum, first_gradient_norm=norm)
            for group in optimizer.param_groups:group['lr']=(C['student_lr'] if cfg['mode']=='student' else C['soft_lr'])*min(1.,(step+1)/10)
            optimizer.step()
            torch.cuda.synchronize(); seconds=time.perf_counter()-began
            if step >= CONFIG['warmup_steps']:
                timings.append(seconds);losses.append(loss_sum)
            print(json.dumps(dict(index=index,step=step,seconds=seconds,loss=loss_sum)),flush=True)
        result.update(status='complete',batch_size=batch_size,timed_step_seconds=timings,median_seconds=statistics.median(timings),
            pairs_per_second=batch_size/statistics.median(timings),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,reference_precompute_seconds=precompute,losses=losses)
        baselines = [i for i,c in enumerate(CONFIG['configs']) if c['mode']==cfg['mode'] and c['microbatch']==2 and c['checkpoint']=='all' and not c['cached_reference']]
        baseline = torch.load(ART/f'gradient-{baselines[0]}.pt',weights_only=True)
        probe = torch.load(ART/f'gradient-{index}.pt',weights_only=True)
        result.update(gradient_cosine=float(torch.nn.functional.cosine_similarity(probe,baseline,dim=0)),gradient_relative_l2=float((probe-baseline).norm()/baseline.norm().clamp_min(1e-12)))
    except torch.OutOfMemoryError as error:
        result.update(status='out_of_memory',error=str(error))
    atomic(ART/f'case-{index}.json',result)


def main():
    from report import render
    ART.mkdir(parents=True,exist_ok=True)
    for index,cfg in enumerate(CONFIG['configs']):
        atomic(RUN/'progress.json',dict(stage='benchmark',case=index,expected=len(CONFIG['configs']),**cfg))
        result=subprocess.run([sys.executable,str(HERE/'benchmark.py'),'worker',str(index)])
        if result.returncode:
            atomic(ART/f'case-{index}.json',dict(index=index,**cfg,status='error',exit_code=result.returncode))
        render()
    atomic(RUN/'progress.json',dict(stage='complete',expected=len(CONFIG['configs'])))
    render()


if __name__=='__main__':
    if len(sys.argv)>1:worker(int(sys.argv[2]))
    else:main()
