"""Local Qwen judge for DSSE behavior evals and saved-answer timing pilots."""
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL = 'Qwen/Qwen3.5-9B'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
BATCH = 32
HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side='left')
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION,
                                                            dtype=torch.bfloat16).cuda().eval()
    return tokenizer, model


@torch.inference_mode()
def evaluate(behavior, rows, batch_size=BATCH, max_tokens=256):
    tokenizer, model = load()
    results = []
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset+batch_size]
        prompts = []
        for row in chunk:
            rubric = behavior['eval_rubric'].replace('{prompt}', row['prompt']).replace('{response}', row['response'])
            prompts.append(tokenizer.apply_chat_template([dict(role='user', content=rubric)],
                          tokenize=False, add_generation_prompt=True, enable_thinking=False))
        inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
        output = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=max_tokens,
                                pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(output[:,inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for text in texts:
            try:
                parsed = json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
                score = float(parsed['score'])
                if score not in (0,.2,.5,1) or not all(isinstance(parsed[k], str) for k in ('label','thinking_str')):
                    raise ValueError('Unexpected verdict')
            except (KeyError,ValueError,TypeError):
                parsed, score = None, None
            results.append(dict(index=len(results), score=score, response=parsed, raw=text))
    return results


def benchmark(behavior, inputs, art):
    import gc
    import random
    from podrun.common import atomic
    order = list(range(len(inputs)))
    random.Random(42).shuffle(order)
    old = next(art.parent.parent.glob('*--qwen35-rejudge-a1/artifacts/verdicts.json'))
    reference = json.loads(old.read_text())
    assert len(reference) == len(inputs)
    evaluate(behavior, [inputs[order[0]]], 1, 4)
    trials = []
    for size in (32,64,96,128,192,256):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        panel = [inputs[i] for i in order[:size]]
        times = []
        oom = False
        try:
            for tokens in (1,17):
                torch.cuda.synchronize()
                start = time.perf_counter()
                evaluate(behavior,panel,size,tokens)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            trials.append(dict(batch=size,oom=True))
            break
        # Model the 316-answer workload, including the partially filled last batch.
        import math
        tokens = sorted(len(load()[0].encode(v['raw'],add_special_tokens=False))+1 for v in reference)
        typical_batch_max = tokens[round((size/(size+1))*(len(tokens)-1))]
        estimate = math.ceil(len(inputs)/size)*(times[0]+max(0,times[1]-times[0])*(typical_batch_max-1)/16)
        trials.append(dict(batch=size,one_token_s=times[0],seventeen_token_s=times[1],
                           predicted_seconds=estimate,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        atomic(art/'screen.json',trials)
    for trial in sorted((t for t in trials if 'predicted_seconds' in t),key=lambda t:t['predicted_seconds']):
        size = trial['batch']
        try:
            start = time.perf_counter()
            verdicts = evaluate(behavior,[inputs[i] for i in order[:size]],size)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
        except torch.cuda.OutOfMemoryError:
            trial['full_oom']=True
            continue
        atomic(art/'validation.json',verdicts)
        valid=sum(v['score'] is not None for v in verdicts)
        matches=sum(v['score']==reference[i]['score'] for i,v in zip(order,verdicts))
        atomic(HERE/'measurements'/(os.environ['PODRUN_ID']+'-batch-pilot.json'),
               dict(trials=trials,validated_batch=size,seconds=elapsed,valid=valid,
                    score_agreement=matches,gpu=torch.cuda.get_device_name()))
        return
    raise RuntimeError('No batch passed full decoding')


def run():
    from huggingface_hub import hf_hub_download
    from podrun.common import atomic
    source = next((HERE/'measurements').glob('*per-answer-a1-dataset.json'))
    previous = json.loads(source.read_text())
    old = Path(previous['artifact_root'])
    cfg = json.loads((HERE/'per-answer.json').read_text())
    art = Path(os.environ['PODRUN_ARTIFACT_DIR'])
    started = time.perf_counter()
    behaviors = hf_hub_download('Transluce/DSSE','behaviors.json',repo_type='dataset',revision=cfg['dataset_revision'])
    behavior = next(b for b in json.loads(Path(behaviors).read_text()) if b['behavior_id']==cfg['behavior_id'])
    methods = ['Base','LoRA','Soft prompt','Verbalization']
    inputs = []
    hashes = {}
    for method in methods:
        raw = (old/(method+'-answers.json')).read_bytes()
        hashes[method] = hashlib.sha256(raw).hexdigest()
        rows = json.loads(raw)
        assert [r['prompt'] for r in rows] == [p['prompt'] for p in behavior['eval_prompts']]
        inputs.extend(rows)
    load()
    torch.cuda.synchronize()
    if '--benchmark' in __import__('sys').argv:
        benchmark(behavior,inputs,art)
        return
    ready = time.perf_counter()
    verdicts = evaluate(behavior,inputs)
    torch.cuda.synchronize()
    done = time.perf_counter()
    atomic(art/'verdicts.json',verdicts)
    n = len(behavior['eval_prompts'])
    summary = {}
    for i,method in enumerate(methods):
        scores = [r['score'] for r in verdicts[i*n:(i+1)*n] if r['score'] is not None]
        summary[method] = dict(valid=len(scores),invalid=n-len(scores),mean=sum(scores)/len(scores) if scores else None)
    atomic(HERE/'measurements'/(os.environ['PODRUN_ID']+'-rejudge.json'),
           dict(model=MODEL,revision=REVISION,source_run=previous['run_id'],source_sha256=hashes,
                batch=BATCH,count=len(inputs),load_and_prepare_seconds=ready-started,
                judging_seconds=done-ready,total_seconds=time.perf_counter()-started,
                gpu=torch.cuda.get_device_name(),scores=summary))

if __name__=='__main__':
    run()
