"""Cache and scoring-batch pilots on the archived seed-42 soft prompt."""
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from run import C, ART, OUT, load_module, write

PREVIOUS = Path('/workspace/projects/verbalizing-subliminal-learning/experiments/2026-11-08-003-length-beam/runs/length-beam-a2/artifacts')


def main():
    import torch
    from salve.objectives import NLLObjective
    from salve.salve import VERBALIZE_PROMPTS, selection_subset, system_header
    from salve.soft_prompt import make_soft_prompt_input, embed_soft_prompt_inputs, sample_with_soft_prompt
    from podrun.activation_checkpointing import Policy
    from performance_report import render

    summary = json.loads((PREVIOUS / 'comparison.json').read_text())
    result = dict(source_commit=os.environ['PODRUN_COMMIT'], complete=False,
                  generation=[], agreement=[], scoring=[], scoring_minibatch_changed=False,
                  search_scoring_minibatch=1, generation_use_cache=True)
    def publish(phase):
        result['phase'] = phase
        write(ART/'performance.json', result)
        write(OUT/'performance.json', result)
        write(ART/'comparison.json', summary)
        write(OUT/'comparison.json', summary)
        write(OUT/'progress.json', dict(phase=phase, time=time.time()))
        render(summary, result)
    publish('loading performance pilot')
    old = ROOT/'fork_experiments/2026-10-28-001-four-rule-qwen35-five'
    model, tok = load_module('cache_model_adapter', old/'model_adapter.py').load(C)
    policy = Policy.from_environment()
    policy.configure_model(model)
    policy.assert_model(model)
    for p in model.parameters():
        p.requires_grad = False
    source = json.loads((HERE/'sources.json').read_text())[0]
    checkpoint = Path(source['path'])/'soft_prompt.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source['checkpoint_sha256']
    z = torch.load(checkpoint, map_location=model.device, weights_only=True)
    result['hardware'] = dict(gpu=torch.cuda.get_device_name(), total_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    result['checkpoint_sha256'] = source['checkpoint_sha256']
    nodes = json.loads((PREVIOUS/'seed-42/bucket128-search.json').read_text())['nodes']
    prefixes = [min((n for n in nodes if n['tokens'] <= 96), key=lambda n: abs(n['tokens']-target)) for target in [0,32,64,96]]

    def measure(fn):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return value, dict(seconds=time.perf_counter()-t0,
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                           incremental_peak_gib=(torch.cuda.max_memory_allocated()-baseline)/2**30)

    for case, node in enumerate(prefixes):
        prompt = VERBALIZE_PROMPTS[case]
        inp = make_soft_prompt_input(tok, prompt['user'], prefill=prompt['prefill']+system_header(tok)+node['text'])
        def sample(cached):
            torch.manual_seed(31415)
            return sample_with_soft_prompt(model, inp, z, 32, 0, set(), use_cache=cached)
        publish(f'cache pilot: prefix {node["tokens"]} tokens')
        # Warm both paths, including recurrent decode kernels, before timing.
        warm = {cached: sample(cached) for cached in [False,True]}
        assert sample(True) == warm[True], 'Cache leaked across calls'
        measurements = {False: [], True: []}
        for repetition in range(3):
            for cached in ([False,True] if repetition % 2 == 0 else [True,False]):
                tokens, metric = measure(lambda: sample(cached))
                assert len(tokens) == 32
                measurements[cached].append(metric)
        result['generation'].append(dict(prefix_tokens=node['tokens'], query=case,
            uncached=measurements[False], cached=measurements[True],
            greedy_token_agreement=sum(a==b for a,b in zip(warm[False],warm[True]))/32))
        # Fixed-token replay: compare logits along the SAME token path, even if
        # BF16 rounding changes a free-running greedy choice at a near tie.
        with torch.no_grad():
            embeds,_,_ = embed_soft_prompt_inputs(model,[inp],z)
            cached_embed=embeds
            cache=None
            checks=[]
            for token in warm[False]:
                plain=model(inputs_embeds=embeds,use_cache=False).logits[:,-1,:].float()
                kwargs=dict(inputs_embeds=cached_embed,use_cache=True)
                if cache is not None:
                    kwargs['past_key_values']=cache
                output=model(**kwargs)
                cache=output.past_key_values
                cached=output.logits[:,-1,:].float()
                lp=torch.log_softmax(plain,-1);lq=torch.log_softmax(cached,-1)
                checks.append(dict(max_abs=float((plain-cached).abs().max()),
                                   kl=float((lp.exp()*(lp-lq)).sum()),
                                   top1=bool(plain.argmax()==cached.argmax())))
                next_embed=model.get_input_embeddings()(torch.tensor([[token]],device=z.device))
                embeds=torch.cat([embeds,next_embed],1);cached_embed=next_embed
                del output, plain, cached, lp, lq
            del embeds, cached_embed, cache, next_embed
        agreement=dict(prefix_tokens=node['tokens'], max_abs=max(c['max_abs'] for c in checks),
                       max_kl=max(c['kl'] for c in checks), mean_kl=statistics.mean(c['kl'] for c in checks),
                       top1_agreement=statistics.mean(c['top1'] for c in checks))
        result['agreement'].append(agreement)
        publish('cache agreement checked')
        if agreement['max_kl'] > .02 or agreement['top1_agreement'] < .90:
            raise RuntimeError('Cached logits exceeded pilot agreement thresholds: '+str(agreement))

    data_path=Path(C['dataset_path'])
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()==C['dataset_sha256']
    data=json.loads(data_path.read_text())
    obj=NLLObjective(model,tok)
    train=obj.prepare([(r['user'],tok.encode(r['response'],add_special_tokens=False)+[tok.eos_token_id]) for r in data['train']])
    subset=selection_subset(train,256,42)
    longest=sorted(subset,key=lambda r:len(r.soft_input.before_prompt)+len(r.soft_input.after_prompt)+len(r.response_ids),reverse=True)[:32]
    winner=summary['results'][0]['methods']['bucket128_best']['text']
    # Full 256-example workload for throughput; longest 32 rows for stress.
    for name, rows in [('representative256',subset),('longest32',longest)]:
        baseline=None
        for batch in [1,2,4,8,16,32,64,128,256]:
            if batch>len(rows):
                break
            publish(f'scoring pilot: {name}, minibatch {batch}')
            error=None
            try:
                # Warm once per layout, then measure three complete evaluations.
                obj.loss_of_text(winner,rows,batch)
                samples=[]; losses=[]
                for _ in range(3):
                    loss,metric=measure(lambda:obj.loss_of_text(winner,rows,batch))
                    samples.append(metric);losses.append(loss)
                if baseline is None:
                    baseline=statistics.mean(losses)
                row=dict(workload=name, examples=len(rows), minibatch=batch, samples=samples,
                         mean_nll=statistics.mean(losses), nll_difference=statistics.mean(losses)-baseline,
                         median_seconds=statistics.median(s['seconds'] for s in samples))
                row['agreement_passed']=abs(row['nll_difference'])<.005
                result['scoring'].append(row)
            except torch.OutOfMemoryError:
                error='CUDA out of memory'
            if error:
                gc.collect();torch.cuda.empty_cache()
                result['scoring'].append(dict(workload=name,minibatch=batch,error=error))
                publish('scoring memory limit reached')
                break
            publish('scoring measurement saved')
    eligible=[r for r in result['scoring'] if r['workload']=='representative256' and r.get('agreement_passed')]
    result['fastest_scoring_minibatch']=min(eligible,key=lambda r:r['median_seconds'])['minibatch']
    result['complete']=True
    publish('performance pilots complete; search minibatch remains 1')


if __name__=='__main__':
    main()
