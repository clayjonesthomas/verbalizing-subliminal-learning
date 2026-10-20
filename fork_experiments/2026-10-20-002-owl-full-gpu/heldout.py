"""Score frozen epoch checkpoints on held-out and fixed training examples."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from salve.salve import load_model
from fork_experiments.owl_full_gpu import run


def encode(rows, tokenizer):
    encoded = []
    for row in rows:
        prompt = [{'role':'system','content':run.student.SYSTEM}, {'role':'user','content':row['prompt']}]
        prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(prompt + [{'role':'assistant','content':row['completion']}], tokenize=True)
        assert full[:len(prefix)] == prefix and len(full) < 512
        encoded.append((full, len(prefix)))
    return encoded


@torch.inference_mode()
def score(model, data, pad):
    model.eval()
    losses, counts = [], []
    for start in range(0, len(data), 8):
        batch = data[start:start+8]
        width = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch),width),pad,dtype=torch.long,device='cuda')
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids,-100)
        for i,(tokens,prefix) in enumerate(batch):
            ids[i,:len(tokens)] = torch.tensor(tokens,device='cuda')
            attention[i,:len(tokens)] = 1
            labels[i,prefix:len(tokens)] = ids[i,prefix:len(tokens)]
        logits = model(input_ids=ids,attention_mask=attention,use_cache=False).logits
        for i in range(len(batch)):
            target=labels[i,1:];mask=target != -100
            loss=F.cross_entropy(logits[i,:-1][mask].float(),target[mask],reduction='sum')
            losses.append(loss.item());counts.append(mask.sum().item())
        del logits
    return dict(token_mean_nll=sum(losses)/sum(counts),example_mean_nll=sum(x/n for x,n in zip(losses,counts))/len(data),
                n_examples=len(data),n_tokens=sum(counts),per_example_nll=[x/n for x,n in zip(losses,counts)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();out=args.results/'heldout';out.mkdir(exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Refusing overwrite'
    source=run.provenance()
    train=run.student.rows(run.HERE/'inputs/owl_train.jsonl')
    validation=run.student.rows(run.HERE/'inputs/validation.jsonl')
    assert len(validation)==500 and not {r['prompt'] for r in train}&{r['prompt'] for r in validation}
    indices=random.Random(20261020).sample(range(len(train)),500)
    manifests={seed:json.loads((args.results/f'owl_seed{seed}/manifest.json').read_text()) for seed in [42,43]}
    free,_=torch.cuda.mem_get_info();assert free>22*2**30, f'Not enough free memory: {free/2**30:.1f} GiB'
    model,tokenizer=load_model(run.student.MODEL)
    assert model.config._commit_hash==manifests[42]['model_revision']
    data={'validation':encode(validation,tokenizer),'train_sample':encode([train[i] for i in indices],tokenizer)}
    metadata=dict(**source,training_sources={s:m['repository_commit'] for s,m in manifests.items()},
                  validation_sha256=run.student.digest(run.HERE/'inputs/validation.jsonl'),train_sha256=run.student.digest(run.HERE/'inputs/owl_train.jsonl'),
                  train_sample_indices=indices,batch_size=8,scoring='Eval mode, no dropout; completion-only tokens including chat-template suffix, token-weighted mean NLL',
                  system_prompt=run.student.SYSTEM,started=time.time())
    run.write(out/'manifest.json',metadata)
    results=[]
    def record(model,seed,epoch,checkpoint=None):
        result=dict(seed=seed,epoch=epoch,metrics={k:score(model,v,tokenizer.pad_token_id) for k,v in data.items()})
        if checkpoint:
            result['adapter_sha256']=run.student.digest(checkpoint/'adapter_model.safetensors')
            result['checkpoint']=str(checkpoint)
        results.append(result);run.write(out/'results.json',results)
        print(json.dumps({k:v for k,v in result.items() if k!='metrics'} | {'nll':{k:v['token_mean_nll'] for k,v in result['metrics'].items()}}),flush=True)
    record(model,None,0)
    for seed in [42,43]:
        checkpoints=sorted((args.results/f'owl_seed{seed}/checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        assert len(checkpoints)==10
        for epoch,checkpoint in enumerate(checkpoints,1):
            adapted=PeftModel.from_pretrained(model,str(checkpoint)).eval()
            record(adapted,seed,epoch,checkpoint)
            model=adapted.unload();del adapted
    run.write(out/'complete.json',dict(status='complete',finished=time.time(),evaluated_points=len(results)))

if __name__=='__main__':
    main()
