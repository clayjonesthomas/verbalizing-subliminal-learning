"""Frozen student-output-trained soft prompt: sentence-wise SALVE readout and hard-text NLL."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

HERE = Path(__file__).resolve().parent
C = json.loads((HERE/'config.json').read_text())
LABEL=os.environ['STYLE_CONDITION']
C=dict(C,**C['conditions'][LABEL],condition=LABEL)
INPUTS=Path(C['input_dir'])
OUT = Path(os.environ['PODRUN_ARTIFACT_DIR'])/LABEL


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def append(name, value):
    with (OUT/name).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')


def progress(stage, **values):
    write('progress.json', dict(stage=stage, time=time.time(), **values))
    p=Path(os.environ['PODRUN_DIR'])/'progress.json'
    p.write_text(json.dumps(dict(condition=LABEL,stage=stage,time=time.time(),**values)))
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def cut_at_sentence(text):
    text = text.replace('</think>', '').replace('<think>', '')
    match = re.search(r'''[.!?]+["'”’]?(?=\s|$)''', text)
    if match is None:
        return text
    end = match.end()
    while end < len(text) and text[end] in ' \t\n':
        end += 1
    return text[:end]


def frontier(children, width, token_cap):
    return sorted((r for r in children if r['tokens'] < token_cap),
                  key=lambda r: (r['score'], r['id']))[:width]


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, StoppingCriteria, StoppingCriteriaList
    from report import render
    OUT.mkdir(parents=True, exist_ok=True)
    write('scientific_config.json', C)
    progress('loading_checkpoint', step=0)
    render()
    checkpoint_path = Path(C['checkpoint'])
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == C['checkpoint_sha256']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['step'] == C['checkpoint_step']
    tok = AutoTokenizer.from_pretrained(C['model'], revision=C['model_revision'], local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(C['model'], revision=C['model_revision'],
        local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    z = checkpoint['z'].to(device='cuda', dtype=model.get_input_embeddings().weight.dtype)
    assert z.shape == (256, model.get_input_embeddings().embedding_dim)
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    embed = lambda ids: model.get_input_embeddings()(torch.tensor(ids, device='cuda'))
    manifest = json.loads((INPUTS/'dataset_manifest.json').read_text())
    data = {}
    for split in ['train', 'validation']:
        path = INPUTS/f'{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['splits'][split]['sha256']
        data[split] = [json.loads(line) for line in path.read_text().splitlines()]
    assert not {r['prompt'] for r in data['train']} & {r['prompt'] for r in data['validation']}
    ids = torch.randperm(len(data['train']), generator=torch.Generator().manual_seed(C['seed'])).tolist()[:C['select_count']]
    data['selection'] = [data['train'][i] for i in ids]
    write('provenance.json', dict(source_commit=os.environ['PODRUN_COMMIT'], checkpoint_sha256=C['checkpoint_sha256'],
        checkpoint_step=checkpoint['step'], training_source_commit=C['training_source_commit'],
        model=C['model'], model_revision=C['model_revision'], selection_row_offsets=ids,
        selection_example_ids=[r['index'] for r in data['selection']], input_manifest=manifest,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), test_used=False, training_performed=False))
    cache = {}

    @torch.inference_mode()
    def score(text, split='selection'):
        key = (text, split)
        if key in cache:
            return cache[key]
        encoded = []
        for row in data[split]:
            prefix = encode(tok.apply_chat_template([{'role':'system','content':text},
                {'role':'user','content':C['user_template'].format(prompt=row['prompt'])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False))
            target = encode(row['response'])+[tok.eos_token_id]
            assert len(prefix)+len(target) <= C['max_train_length']
            encoded.append((prefix, target))
        total = 0.
        denominator = sum(len(target) for _, target in encoded)
        for offset in range(0, len(encoded), C['eval_batch_size']):
            batch = encoded[offset:offset+C['eval_batch_size']]
            lengths = [len(prefix)+len(target) for prefix,target in batch]
            width = max(lengths)
            tokens = torch.tensor([prefix+target+[tok.eos_token_id]*(width-n)
                for (prefix,target),n in zip(batch,lengths)], device='cuda')
            mask = (torch.arange(width,device='cuda')[None,:] < torch.tensor(lengths,device='cuda')[:,None]).long()
            h = model.model(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
            hidden = torch.cat([h[i,len(prefix)-1:len(prefix)-1+len(target)] for i,(prefix,target) in enumerate(batch)])
            targets = torch.tensor([token for _,target in batch for token in target], device='cuda')
            for start in range(0,len(targets),128):
                total += float(F.cross_entropy(model.lm_head(hidden[start:start+128]).float(),targets[start:start+128],reduction='sum'))
        result = total/denominator
        assert math.isfinite(result)
        cache[key] = result
        return result

    @torch.inference_mode()
    def generate(query_index, prefix='', greedy=False):
        q = C['queries'][query_index]
        marker = 'SOFT_SYSTEM_VECTOR_PLACEHOLDER'
        template = tok.apply_chat_template([{'role':'system','content':marker},{'role':'user','content':q['user']}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert template.count(marker) == 1
        before,after = template.split(marker)
        x = torch.cat([embed(encode(before)),z,embed(encode(after)),embed(encode(q['prefill']+prefix))])[None]
        class Stop(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                return bool(q['stop']) and q['stop'] in tok.decode(input_ids[0,-12:],skip_special_tokens=True)
        sampling = {} if greedy else dict(temperature=C['temperature'],top_p=C['top_p'],top_k=C['top_k'])
        output = model.generate(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),
            do_sample=not greedy, max_new_tokens=256 if greedy else C['extension_tokens'],use_cache=True,
            pad_token_id=tok.eos_token_id,stopping_criteria=StoppingCriteriaList([Stop()]),**sampling)[0].tolist()
        raw = tok.decode(output,skip_special_tokens=True)
        return raw, raw.split(q['stop'],1)[0] if q['stop'] else raw

    progress('greedy_reference')
    raw, text = generate(0,greedy=True)
    greedy = dict(text=text.strip(),raw=raw,selection_nll=score(text.strip()),validation_nll=score(text.strip(),'validation'))
    write('greedy.json',greedy)
    torch.manual_seed(C['seed'])
    rng = random.Random(C['seed'])
    bag = []
    nodes = [dict(id=0,text='',tokens=0,depth=0)]
    active = nodes[:]
    best = None
    attempted = 0
    for iteration in range(1,C['rounds']+1):
        if not active:
            break
        children = []
        for parent in active:
            draws = C['n_beams']*C['branching'] if parent['id']==0 else C['branching']
            for _ in range(draws):
                if not bag:
                    bag.extend(rng.sample(range(len(C['queries'])),len(C['queries'])))
                query = bag.pop()
                attempted += 1
                progress('beam_generation',step=attempted,round=iteration,scored=len(nodes)-1)
                raw,continuation = generate(query,parent['text'])
                sentence = cut_at_sentence(continuation)
                text = parent['text']+sentence
                tokens = len(encode(text))
                if not sentence.strip() or tokens>C['prompt_token_cap']:
                    append('rejected.jsonl',dict(attempt=attempted,round=iteration,parent=parent['id'],query=query,raw=raw,reason='empty_or_overlength'))
                    continue
                progress('beam_scoring',step=attempted,round=iteration,scored=len(nodes)-1)
                child = dict(id=len(nodes),parent=parent['id'],depth=parent['depth']+1,round=iteration,
                    attempt=attempted,query=query,text=text,sentence=sentence,raw=raw,tokens=tokens,score=score(text))
                nodes.append(child)
                children.append(child)
                append('candidates.jsonl',child)
                if best is None or child['score']<best['score']:
                    best=child
                append('metrics.jsonl',dict(step=len(nodes)-1,round=iteration,nll=child['score'],best_nll=best['score'],time=time.time()))
                write('summary.json',dict(status='searching',round=iteration,scored=len(nodes)-1,attempted=attempted,best=best))
                if (len(nodes)-1)%8==0:
                    render()
        active = frontier(children,C['n_beams'],C['prompt_token_cap'])
        write('beam_state.json',dict(round=iteration,active_ids=[r['id'] for r in active],best=best,nodes=nodes))
        render()
    assert best is not None, 'No nonempty candidates generated'
    top = []
    seen = set()
    for row in sorted(nodes[1:],key=lambda r:(r['score'],r['id'])):
        if row['text'] not in seen:
            top.append(row);seen.add(row['text'])
        if len(top)==C['validation_top_k']:
            break
    validated=[]
    for row in top:
        progress('heldout_validation',step=row['id'],completed=len(validated),total=len(top))
        validated.append(dict(row,validation_nll=score(row['text'],'validation')))
        write('validations.json',validated)
        render()
    write('summary.json',dict(status='complete',round=iteration,scored=len(nodes)-1,attempted=attempted,
        best=validated[0],greedy=greedy,top=validated,selection=C['selection'],test_used=False,
        source_commit=os.environ['PODRUN_COMMIT'],checkpoint_step=C['checkpoint_step']))
    progress('complete',step=len(nodes)-1)
    render()


if __name__ == '__main__':
    main()
