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
def evaluate(behavior, rows):
    tokenizer, model = load()
    results = []
    for offset in range(0, len(rows), BATCH):
        chunk = rows[offset:offset+BATCH]
        prompts = []
        for row in chunk:
            rubric = behavior['eval_rubric'].replace('{prompt}', row['prompt']).replace('{response}', row['response'])
            prompts.append(tokenizer.apply_chat_template([dict(role='user', content=rubric)],
                          tokenize=False, add_generation_prompt=True, enable_thinking=False))
        inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
        output = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=256,
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
