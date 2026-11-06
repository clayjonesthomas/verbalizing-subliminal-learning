"""Four greedy SALVE queries, selected on fixed training-subset hard-text NLL."""
import os,json,math
import torch
import torch.nn.functional as F
from transformers import StoppingCriteria,StoppingCriteriaList

def run_best_four(model,tok,unused_z,OUT,INPUTS,C,write,progress):
    model.eval()
    model.gradient_checkpointing_disable()
    checkpoint=torch.load(OUT/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    z=checkpoint['z'].to(device='cuda',dtype=model.get_input_embeddings().weight.dtype)
    import hashlib
    C=dict(C,checkpoint_sha256=hashlib.sha256((OUT/'checkpoints/best.pt').read_bytes()).hexdigest())
    encode=lambda text:tok.encode(text,add_special_tokens=False)
    embed=lambda ids:model.get_input_embeddings()(torch.tensor(ids,device='cuda'))
    data={split:[json.loads(line) for line in (INPUTS/(split+'.jsonl')).read_text().splitlines()] for split in ['train','validation']}
    indices=torch.randperm(len(data['train']),generator=torch.Generator().manual_seed(C['seed'])).tolist()[:C['select_count']]
    data['selection']=[data['train'][i] for i in indices]
    write('best_four_selection.json',dict(row_offsets=indices,example_ids=[r['index'] for r in data['selection']]))
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
            do_sample=not greedy, max_new_tokens=128 if greedy else C['extension_tokens'],use_cache=True,
            pad_token_id=tok.eos_token_id,stopping_criteria=StoppingCriteriaList([Stop()]),**sampling)[0].tolist()
        raw = tok.decode(output,skip_special_tokens=True)
        return raw, raw.split(q['stop'],1)[0] if q['stop'] else raw

    from clean_readout import clean_verbalization
    candidates = []
    for query_index, query in enumerate(C['queries']):
        progress('best_of_four_generation', completed=query_index, total=4)
        raw, _ = generate(query_index, greedy=True)
        text = clean_verbalization(raw, query['stop'])
        row = dict(query_index=query_index, query=query['user'], raw=raw, text=text,
                   selection_nll=score(text), validation_nll=score(text, 'validation'))
        candidates.append(row)
        write('candidates.json', candidates)
    selected = min(candidates, key=lambda r: (r['selection_nll'], r['query_index']))
    result = dict(candidates=candidates, selected=selected, source_commit=os.environ['PODRUN_COMMIT'],
        checkpoint_sha256=C['checkpoint_sha256'], checkpoint_step=checkpoint['step'],
        generation='greedy, 128-token cap, four SALVE extraction queries',
        selection='minimum NLL on the same fixed 256 training stories; ties prefer lower query index',
        validation='all 1000 validation stories, diagnostic only', test_used=False)
    write('best_of_four.json', result)
