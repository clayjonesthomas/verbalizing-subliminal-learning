import json,numpy as np,torch
from pathlib import Path
C=Path('~/.cache/research-infra/animal-cosine').expanduser();P=Path(__file__).parent;D=json.loads((P/'results.json').read_text());torch.set_num_threads(8)
e=torch.from_file(str(C/'embeddings.bf16'),size=152064*3584,dtype=torch.bfloat16).reshape(152064,3584)
normalize=lambda a:a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-30)
d=normalize(e[list(D['canonical_token_ids'].values())].double().numpy());u=normalize(d.mean(0))
checks=[]
for animal,seed,mode in [('cat',42,'raw'),('eagle',50,'residual'),('owl',56,'raw')]:
 r=next(x for x in D['rows'] if x['animal']==animal and x['seed']==seed);j=r[mode]['max_position']-1;t=D['canonical_token_ids'][animal]
 z=torch.load(C/f'{animal}-{seed}.pt',weights_only=True,map_location='cpu')['z'][j].double().numpy();target=e[t].double().numpy()
 def tr(a):
  a=normalize(a)
  if mode=='residual':a=normalize(a-np.sum(a*u,axis=-1,keepdims=True)*u)
  return a
 z=tr(z);threshold=float(z@tr(target));rank=1
 for start in range(0,152064,2048):
  b=e[start:start+2048].double().numpy();nonzero=np.linalg.norm(b,axis=1)>1e-6;sc=tr(b)@z;higher=(sc>threshold)&nonzero
  if start<=t<start+len(b):higher[t-start]=False
  rank+=int(higher.sum())
 assert abs(threshold-r[mode]['max_cosine'])<1e-6
 assert rank==r[mode]['rank_at_max'],(rank,r[mode]['rank_at_max'])
 checks.append(dict(animal=animal,seed=seed,mode=mode,rank=rank,cosine=threshold))
print(checks)
(P/'validation.json').write_text(json.dumps(dict(independent_float64_numpy_checks=checks,checkpoint_hashes_matched_historical_best4=60),indent=2))
