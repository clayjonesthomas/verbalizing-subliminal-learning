"""SALVE's training and search as separately reportable, resumable stages."""
import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import torch
from salve.salve import SoftPromptConfig,SearchConfig,load_model
from salve.objectives import NLLObjective
from subliminal_learning.datasets import load_splits
from subliminal_learning.animal_prompts import trait_prompt
from fork_experiments import EXPERIMENTS
from fork_experiments.owl_full.instrumented import train_steps,beam_search
from fork_experiments.owl_full.objective import EfficientNLLObjective
from fork_experiments.owl_full.performance import enable_cache,cached_sample
from fork_experiments.owl_full.provenance import committed_source
from fork_experiments.owl_full.trajectory import Trajectory,preserve_rng,save
from salve.soft_prompt import make_soft_prompt_input,sample_with_soft_prompt
from salve.salve import VERBALIZE_PROMPTS,stop_token_ids

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/salve_seed42'
DATA=HERE/'results/numbers_seed42/splits'

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','search']);parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
    OUT=HERE/'results'/f'salve_seed{args.seed}'
    source=committed_source('finetuned_teacher_owl')
    repo=HERE.parents[1]
    for p in (repo/'fork_experiments/2026-10-19-001-owl-salve-largo').glob('*.py'):
        assert p.read_bytes()==subprocess.check_output(['git','show',f"{source['repository_commit']}:{p.relative_to(repo)}"])
    OUT.mkdir(exist_ok=True,parents=True)
    model,tokenizer=load_model('Qwen/Qwen2.5-7B-Instruct')
    reference=NLLObjective(model,tokenizer);obj=EfficientNLLObjective(model,tokenizer)
    splits=load_splits(DATA);train,val=obj.prepare(splits['train']),obj.prepare(splits['validation'])
    assert len(train)==10000 and len(val)==500
    soft=SoftPromptConfig(mini_batch_size=16);search=SearchConfig(mini_batch_size=64)
    trajectory=Trajectory(obj,val,OUT,search.mini_batch_size)
    if args.stage=='search':
        prior=json.loads((OUT/'training_result.json').read_text())['diagnostic']
        trajectory.started -= prior['elapsed_seconds']+prior['evaluation_seconds']
    manifest={**source,'stage':args.stage,'model_revision':model.config._commit_hash,
        'dataset_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.jsonl')},
        'soft_config':dataclasses.asdict(soft),'search_config':dataclasses.asdict(search),'seed':args.seed,
        'teacher_description':'fine-tuned owl teacher under neutral system prompt; recovery model is frozen original Qwen',
        'started_unix':time.time()}
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    E=model.get_input_embeddings().weight
    z=torch.randn(soft.n_tokens,E.shape[1],device=E.device,dtype=E.dtype)*E.std()
    if args.stage=='train':
        assert not (OUT/'soft_prompt.pt').exists(),'Training already completed'
        # Benchmark real soft-prompt gradients with unchanged effective batch 16.
        timings=[]
        with preserve_rng():
            for batch in (8,16):
                zz=z.detach().clone().requires_grad_(True)
                torch.cuda.synchronize();start=time.monotonic()
                loss=obj.loss(zz,train[:16],backward=True,mini_batch_size=batch)
                torch.cuda.synchronize()
                timings.append({'batch':batch,'seconds':time.monotonic()-start,'loss':loss,'gradient_finite':bool(torch.isfinite(zz.grad).all())})
                del zz
            assert all(t['gradient_finite'] for t in timings)
        soft.mini_batch_size=min(timings,key=lambda t:t['seconds'])['batch']
        manifest.update(soft_config=dataclasses.asdict(soft),training_benchmark=timings)
        save(OUT/'train_manifest.json',manifest)
        baselines={name:obj.loss_of_text(text,val,64) for name,text in [('default',None),('empty',''),('prompted_owl_reference',trait_prompt('owl'))]}
        initial=obj.loss(z,val,mini_batch_size=16)
        save(OUT/'baseline_nll.json',dict(text=baselines,initial_soft=initial))
        def checkpoint(step,current,history):
            save(OUT/'training_history.json',history)
            torch.save({'z':current.detach().cpu(),'step':step},OUT/'latest_soft.pt')
            save(OUT/'progress.json',{'stage':'training','step':step,'total':2500,'latest_loss':history['train_loss'][-1] if history['train_loss'] else None})
        start=time.monotonic()
        z,history=train_steps(obj,z,train,soft,2500,on_step=checkpoint)
        assert all(math.isfinite(x) for x in history['train_loss']+history['grad_norm'])
        torch.save({'z':z.cpu(),'history':history,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},OUT/'soft_prompt.pt')
        final=obj.loss(z,val,mini_batch_size=16)
        # Diagnostic readout only; does not select the trained checkpoint.
        diagnostic=trajectory.record('salve_training',2500,z=z)
        save(OUT/'training_result.json',{'status':'complete','steps':2500,'epochs':4,'initial_soft_nll':initial,'final_soft_nll':final,
            'baseline_nll':baselines,'mean_first_100_train_loss':sum(history['train_loss'][:100])/100,
            'mean_last_100_train_loss':sum(history['train_loss'][-100:])/100,'diagnostic':diagnostic,'elapsed_seconds':time.monotonic()-start})
        print('TRAINING COMPLETE',flush=True)
    else:
        assert (OUT/'training_result.json').exists(),'Training report must be complete first'
        assert not (OUT/'result.json').exists(),'Search already complete'
        saved=torch.load(OUT/'soft_prompt.pt',map_location='cpu',weights_only=False);z=saved['z'].to(model.device)
        # Check faster text scoring against reference before using it in search.
        checks=[]
        for text in ('',trait_prompt('owl')):
            ref=reference.loss_of_text(text,val[:32],16);fast=obj.loss_of_text(text,val[:32],16)
            assert abs(ref-fast)<1e-4,(ref,fast)
            checks.append({'text':text,'reference':ref,'optimized':fast})
        save(OUT/'scoring_checks.json',checks)
        # KV caching avoids recomputing the entire soft-prefix for every generated token.
        # Preserve policy and document possible BF16 stochastic-trajectory changes.
        inp=make_soft_prompt_input(tokenizer,VERBALIZE_PROMPTS[0]['user'],prefill=VERBALIZE_PROMPTS[0]['prefill'])
        stops=stop_token_ids(model,tokenizer)
        a=sample_with_soft_prompt(model,inp,z,32,0,stops)
        b=cached_sample(model,inp,z,32,0,stops)
        cache_matches=(a==b)
        manifest['cached_decoding_greedy_check']=cache_matches
        manifest['trajectory_evaluation']='Every search iteration, with RNG isolation; clock continues training and excludes inter-stage pause.'
        manifest['training_source_commit']=json.loads((OUT/'train_manifest.json').read_text())['repository_commit']
        save(OUT/'search_manifest.json',manifest)
        if cache_matches:enable_cache()
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        def iteration(step,nodes,best):
            save(OUT/'search_nodes.json',nodes)
            trajectory.record('salve_search',step,prompt=best['text'],selection_nll=best['score'],n_candidates=len(nodes)-1)
            save(OUT/'progress.json',{'stage':'search','iteration':step,'total':12,'candidates':len(nodes)-1,'best':best})
        start=time.monotonic()
        nodes,best=beam_search(obj,z,train,search,args.seed,on_iteration=iteration)
        save(OUT/'recovered.json',best)
        evaluations={name:trajectory.evaluate(text) for name,text in [('recovered',best['text']),('default',None),('empty',''),('prompted_owl_reference',trait_prompt('owl'))]}
        save(OUT/'result.json',{'status':'complete','prompt':best['text'],'selection_nll':best['score'],'candidates':len(nodes)-1,
             'evaluations':evaluations,'search_seconds':time.monotonic()-start,'repository_commit':source['repository_commit']})
        print('SEARCH COMPLETE',best['text'],flush=True)

if __name__=='__main__':main()
