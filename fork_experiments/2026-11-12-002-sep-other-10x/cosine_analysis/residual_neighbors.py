"""Full-vocabulary neighbors after projecting out the common digit direction."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument('--cache-dir', type=Path, default=Path('~/.cache/research-infra/sep-cosine').expanduser())
p.add_argument('--output', type=Path)
args = p.parse_args()
c = args.cache_dir
torch.set_num_threads(8)
manifest = sorted(json.loads((c/'manifest.json').read_text()), key=lambda r:(r['student'], r['seed']))
assert len(manifest) == 12
meta = json.loads((c/'sep-embed-header.json').read_text())['model.language_model.embed_tokens.weight']
vocab, width = meta['shape']
emb = torch.from_file(str(c/'sep-qwen35-embeddings.bf16'), size=vocab*width, dtype=torch.bfloat16).reshape(vocab,width)
tok = Tokenizer.from_file(str(c/'sep-qwen35-tokenizer.json'))
digit_ids = [tok.encode(str(i),add_special_tokens=False).ids for i in range(10)]
assert all(len(ids)==1 for ids in digit_ids)
digit_ids = torch.tensor([ids[0] for ids in digit_ids])
norm = lambda x: torch.nn.functional.normalize(x,dim=1)
digits = norm(emb[digit_ids].float())
direction = digits.mean(0); direction = direction/direction.norm()
def project(x):
    x = norm(x)
    r = x - (x@direction)[:,None]*direction
    assert r.norm(dim=1).min() > 1e-6, 'Undefined residual cosine for near-zero vector'
    return norm(r)
soft = []
for r in manifest:
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    x=torch.load(r['path'],map_location='cpu',weights_only=True).float()
    assert x.shape==(128,width)
    soft.append(x)
z = project(torch.cat(soft))
assert (z@direction).abs().max()<1e-5
# Check the previously computed digit-only residual cosines agree.
prior = json.loads((c/'digit_distribution.json').read_text())
expected = torch.tensor([pos['residual_digit_cosines'] for r in prior['rows'] for pos in r['per_position']])
assert torch.allclose(z@project(digits).T, expected, atol=1e-6, rtol=1e-5)
values = torch.full((len(z),10),-torch.inf)
ids = torch.zeros((len(z),10),dtype=torch.long)
for start in range(0,vocab,8192):
    scores = z@project(emb[start:start+8192].float()).T
    v, ix = scores.topk(min(10,scores.shape[1]),dim=1)
    merged = torch.cat([values,v],dim=1)
    values, keep = merged.topk(10,dim=1)
    ids = torch.gather(torch.cat([ids,ix+start],dim=1),1,keep)
prefixes = {'s1':'290','s2':'024','s3':'819','s4':'782'}
rows = []
for i,r in enumerate(manifest):
    trigger = {int(digit_ids[int(d)]) for d in prefixes[r['student']]}
    positions=[]
    for j in range(128):
        token_ids=ids[i*128+j].tolist()
        hits=[tok.id_to_token(t) for t in token_ids if t in trigger]
        positions.append(dict(position=j+1,trigger_digits_in_top10=hits,top10=[dict(token_id=t,token=tok.id_to_token(t),cosine=v) for t,v in zip(token_ids,values[i*128+j].tolist())]))
    row=dict(student=r['student'],seed=r['seed'],prefix=prefixes[r['student']],soft_sha256=r['sha256'],trigger_top10_positions=sum(bool(p['trigger_digits_in_top10']) for p in positions),any_digit_top10_positions=sum(any(x['token_id'] in digit_ids.tolist() for x in p['top10']) for p in positions),positions=positions)
    rows.append(row)
    print(r['student'],r['seed'],row['trigger_top10_positions'],'trigger positions;',row['any_digit_top10_positions'],'any-digit positions')
output=dict(model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',embedding_name='model.language_model.embed_tokens.weight',vocab_size=vocab,soft_tokens=128,definition='Mean of ten normalized digit input embeddings defines one common unit direction. Project it out of every soft vector and all vocabulary vectors, renormalize, and rank full-vocabulary residual cosine. Count each position once if any digit of its setting-specific trigger is among its ten nearest neighbors. Digit order and full trigger recovery are not tested.',rows=rows)
(args.output or c/'residual_neighbors.json').write_text(json.dumps(output,indent=2)+'\n')
