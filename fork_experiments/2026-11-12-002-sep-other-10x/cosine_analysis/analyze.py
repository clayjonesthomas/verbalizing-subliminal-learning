"""Compare selected soft prompts to the pinned model input embeddings, on CPU."""
import json,math,hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
import argparse
p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=Path,default=Path('~/.cache/research-infra/sep-cosine').expanduser());args=p.parse_args()
CACHE=args.cache_dir;torch.set_num_threads(8)
manifest=sorted(json.loads((CACHE/'manifest.json').read_text()),key=lambda r:(r['student'],r['seed']))
tok=Tokenizer.from_file(str(CACHE/'sep-qwen35-tokenizer.json'))
meta=json.load(open(CACHE/'sep-embed-header.json'))['model.language_model.embed_tokens.weight']
vocab,width=meta['shape'];path=CACHE/'sep-qwen35-embeddings.bf16'
assert path.stat().st_size==vocab*width*2
emb=torch.from_file(str(path),shared=False,size=vocab*width,dtype=torch.bfloat16).reshape(vocab,width)
soft=[]
for r in manifest:
 assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
 z=torch.load(r['path'],map_location='cpu',weights_only=True).float();assert z.shape==(128,width);soft.append(z)
z=torch.cat(soft);zn=torch.nn.functional.normalize(z,dim=1)
digit_ids=[tok.encode(str(i),add_special_tokens=False).ids for i in range(10)];assert all(len(x)==1 for x in digit_ids)
digit_ids=torch.tensor([x[0] for x in digit_ids]);dn=torch.nn.functional.normalize(emb[digit_ids].float(),dim=1)
dc=zn@dn.T
# For each prompt and digit, rank that digit against all vocabulary at its max-cosine position.
positions=[];thresholds=[]
for i in range(12):
 best=dc[i*128:(i+1)*128].argmax(0)
 for d in range(10):positions.append(i*128+int(best[d]));thresholds.append(float(dc[positions[-1],d]))
positions=torch.tensor(positions);thresholds=torch.tensor(thresholds)
ranks=torch.ones(120,dtype=torch.int64);topvalues=torch.full((1536,10),-math.inf);topids=torch.zeros((1536,10),dtype=torch.int64)
for start in range(0,vocab,8192):
 end=min(vocab,start+8192);en=torch.nn.functional.normalize(emb[start:end].float(),dim=1);sim=zn@en.T
 # Exclude target token itself from rank counts (float roundoff between matrix shapes).
 comparisons=sim[positions]>thresholds[:,None]
 for j in range(120):
  token=int(digit_ids[j%10])
  if start<=token<end:comparisons[j,token-start]=False
 ranks+=comparisons.sum(1)
 values,indices=sim.topk(min(10,end-start),dim=1);indices+=start
 merged=torch.cat([topvalues,values],1);topvalues,ix=merged.topk(10,dim=1)
 topids=torch.gather(torch.cat([topids,indices],1),1,ix)
 print('vocabulary',end,'/',vocab,flush=True)
prefixes={'s1':'290','s2':'024','s3':'819','s4':'782'};results=[]
for i,r in enumerate(manifest):
 digits=[]
 for d in range(10):
  p=int(positions[i*10+d]);digits.append(dict(digit=str(d),token_id=int(digit_ids[d]),trigger_digit=str(d) in prefixes[r['student']],max_cosine=float(dc[p,d]),soft_position=p-i*128+1,vocab_rank=int(ranks[i*10+d]),nearest_tokens=[dict(token_id=int(t),token=tok.id_to_token(int(t)),cosine=float(c)) for t,c in zip(topids[p,:5],topvalues[p,:5])]))
 slots=[]
 for p in range(i*128,(i+1)*128):slots.append(dict(position=p-i*128+1,nearest_token_id=int(topids[p,0]),nearest_token=tok.id_to_token(int(topids[p,0])),cosine=float(topvalues[p,0])))
 results.append(dict(student=r['student'],seed=r['seed'],prefix=prefixes[r['student']],soft_sha256=r['sha256'],digits=digits,nearest_by_slot=slots,trigger_digit_top10_slots=sum(any(int(t) in [int(digit_ids[int(d)]) for d in prefixes[r['student']]] for t in row) for row in topids[i*128:(i+1)*128])))
output=dict(model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',embedding_name='model.language_model.embed_tokens.weight',vocab_size=vocab,soft_tokens=128,position_indexing='1-based',results=results)
(CACHE/'results.json').write_text(json.dumps(output,indent=2)+'\n')
for r in results:
 print(r['student'],r['seed'],'trigger',[(d['digit'],round(d['max_cosine'],4),d['soft_position'],d['vocab_rank']) for d in r['digits'] if d['trigger_digit']], 'best digit',max(r['digits'],key=lambda x:x['max_cosine'])['digit'],'top10 slots',r['trigger_digit_top10_slots'])
