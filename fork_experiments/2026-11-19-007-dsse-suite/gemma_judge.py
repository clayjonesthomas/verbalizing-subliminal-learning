"""Time Gemma verdict batches, then rejudge one immutable DSSE run."""
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from podrun.common import atomic

HERE = Path(__file__).resolve().parent
MODEL = 'google/gemma-4-31B-it'
REVISION = '842da3794eaa0b77d5f08bae87a17459d91ff475'
LABELS = {'no_match':0., 'weak_ambiguous':.2, 'partial_match':.5, 'strong_match':1.}


def verdict(text):
    try:
        result = json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
        if LABELS[result['label']] == float(result['score']) and isinstance(result['thinking_str'], str):
            return result
    except (ValueError, KeyError, TypeError):
        pass
    return None


def run():
    start = time.perf_counter()
    cfg = json.loads((HERE/'per-answer.json').read_text())
    previous = json.loads(next((HERE/'measurements').glob('*per-answer-a1-dataset.json')).read_text())
    expected = json.loads(next((HERE/'measurements').glob('*qwen35-rejudge-a1-rejudge.json')).read_text())
    old = Path(previous['artifact_root'])
    art = Path(os.environ['PODRUN_ARTIFACT_DIR'])
    behavior_file = hf_hub_download('Transluce/DSSE','behaviors.json',repo_type='dataset',revision=cfg['dataset_revision'])
    behavior = next(b for b in json.loads(Path(behavior_file).read_text()) if b['behavior_id']==cfg['behavior_id'])
    tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,padding_side='left')
    model = Gemma4ForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval()
    inputs, metadata = [], []
    methods = ['Base','LoRA','Soft prompt','Verbalization']
    for method in methods:
        raw = (old/(method+'-answers.json')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected['source_sha256'][method]
        answers=json.loads(raw)
        judges=json.loads((old/(method+'-judges.json')).read_text())
        assert len(answers)==len(judges)==len(behavior['eval_prompts'])
        for i,(a,j) in enumerate(zip(answers,judges)):
            assert j['index']==i and a['prompt']==behavior['eval_prompts'][i]['prompt']
            prompt=behavior['eval_rubric'].replace('{prompt}',a['prompt']).replace('{response}',a['response'])
            inputs.append(tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
                                                  return_dict=False,enable_thinking=False,add_generation_prompt=True))
            metadata.append(dict(method=method,prompt=a['prompt'],answer=a['response'],gpt=j['response']))
    @torch.inference_mode()
    def generate(indices):
        batch=tok.pad({'input_ids':[inputs[i] for i in indices]},padding=True,return_tensors='pt').to(model.device)
        result=model.generate(**batch,do_sample=True,temperature=1.,top_p=.95,top_k=64,max_new_tokens=256,
                              use_cache=True)
        return tok.batch_decode(result[:,batch.input_ids.shape[1]:],skip_special_tokens=True)

    torch.manual_seed(42)
    generate([max(range(len(inputs)),key=lambda i:len(inputs[i]))])
    ready=time.perf_counter()
    order=list(range(len(inputs)))
    random.Random(42).shuffle(order)
    trials=[]
    for size in (1,2,4,8,12,16,24,32,48,64,96,128):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        try:
            texts=generate(order[:size])
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(batch=size,oom=True))
            break
        elapsed=time.perf_counter()-t
        trials.append(dict(batch=size,seconds=elapsed,rate=size/elapsed,valid=sum(verdict(x) is not None for x in texts),
                           peak_gib=torch.cuda.max_memory_reserved()/2**30))
        atomic(art/'pilot.json',trials)
    gc.collect()
    torch.cuda.empty_cache()
    candidates=sorted((r for r in trials if 'rate' in r),key=lambda r:-r['rate'])
    assert candidates
    batch=candidates[0]['batch']
    torch.manual_seed(42)
    t=time.perf_counter()
    outputs=[]
    used=[]
    while len(outputs)<len(inputs):
        offset=len(outputs)
        try:
            outputs.extend(generate(list(range(offset,min(offset+batch,len(inputs))))))
            used.append(min(batch,len(inputs)-offset))
        except torch.cuda.OutOfMemoryError:
            if batch==1:raise
            batch=max(1,batch//2)
        gc.collect()
    seconds=time.perf_counter()-t
    atomic(art/'outputs.json',outputs)
    comparisons=[]
    summary={m:dict(valid=0,invalid=0,same=0,higher=0,lower=0,gpt_sum=0.,gemma_sum=0.) for m in methods}
    for meta,text in zip(metadata,outputs):
        v=verdict(text);s=summary[meta['method']]
        if v is None:
            s['invalid']+=1
            continue
        a,b=meta['gpt']['score'],float(v['score'])
        s['valid']+=1;s['gpt_sum']+=a;s['gemma_sum']+=b
        s['same' if a==b else 'higher' if b>a else 'lower']+=1
        if a!=b:comparisons.append(dict(**meta,gemma=v))
    comparisons.sort(key=lambda r:-abs(r['gpt']['score']-float(r['gemma']['score'])))
    atomic(HERE/'measurements'/(os.environ['PODRUN_ID']+'-gemma.json'),
           dict(source_run=previous['run_id'],source_sha256=expected['source_sha256'],revision=REVISION,
                trials=trials,selected=candidates[0]['batch'],batches=used,seconds=seconds,
                preparation=ready-start,pilot_seconds=t-ready,total_seconds=time.perf_counter()-start,
                gpu=torch.cuda.get_device_name(),scores=summary,disagreements=comparisons))

if __name__=='__main__':
    run()
