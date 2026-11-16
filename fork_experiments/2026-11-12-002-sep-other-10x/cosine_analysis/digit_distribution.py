import torch,json,statistics
from pathlib import Path
from tokenizers import Tokenizer
import argparse
p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=Path,default=Path('~/.cache/research-infra/sep-cosine').expanduser());args=p.parse_args()
C=args.cache_dir;torch.set_num_threads(8)
manifest=sorted(json.loads((C/'manifest.json').read_text()),key=lambda r:(r['student'],r['seed']))
e=torch.from_file(str(C/'sep-qwen35-embeddings.bf16'),size=248320*4096,dtype=torch.bfloat16).reshape(248320,4096)
z=torch.cat([torch.load(r['path'],weights_only=True,map_location='cpu').float() for r in manifest]); z=torch.nn.functional.normalize(z,dim=1)
d=torch.nn.functional.normalize(e[15:25].float(),dim=1);cos=z@d.T;best,ids=cos.max(1)
rank=torch.ones(len(z),dtype=torch.long)
for start in range(0,len(e),8192):
 en=torch.nn.functional.normalize(e[start:start+8192].float(),dim=1);scores=z@en.T
 cmp=scores>best[:,None]
 if start==0:cmp[torch.arange(len(z)),ids+15]=False
 rank+=cmp.sum(1)
pair=d@d.T;off=pair[~torch.eye(10,dtype=torch.bool)]
mean=d.mean(0);direction=mean/mean.norm()
# Project away the common digit direction in both soft vectors and digit vectors.
dr=d-(d@direction)[:,None]*direction
zr=z-(z@direction)[:,None]*direction
res=torch.nn.functional.normalize(zr,dim=1)@torch.nn.functional.normalize(dr,dim=1).T
prefix={'s1':'290','s2':'024','s3':'819','s4':'782'}
rows=[]
for i,r in enumerate(manifest):
 b=best[i*128:(i+1)*128];rk=rank[i*128:(i+1)*128];dc=cos[i*128:(i+1)*128];rc=res[i*128:(i+1)*128]
 quantiles={str(q):float(torch.quantile(b,q)) for q in [0.,.25,.5,.75,.9,.95,1.]}
 rows.append(dict(student=r['student'],seed=r['seed'],quantiles=quantiles,above={str(t):int((b>=t).sum()) for t in [.1,.15,.2,.25]},number_rank_le={str(t):int((rk<=t).sum()) for t in [1,10,100,1000]},per_position=[dict(position=j+1,max_digit_cosine=float(b[j]),best_digit=str(int(ids[i*128+j])),best_digit_vocab_rank=int(rk[j]),residual_digit_cosines=rc[j].tolist()) for j in range(128)],residual_max_per_digit=rc.max(0).values.tolist()))
out=dict(definition='At each soft position, maximum cosine against the ten single digit input embeddings; ranks against all 248320 input embeddings. Thresholds are descriptive, not statistical cutoffs.',digit_pairwise_cosines=pair.tolist(),off_diagonal_summary=dict(min=float(off.min()),mean=float(off.mean()),max=float(off.max())),common_direction_squared_fraction=(d@direction).square().tolist(),residual_definition='Project both normalized soft vectors and normalized digit embeddings perpendicular to the mean normalized digit direction, then renormalize and compute cosine.',rows=rows)
(C/'digit_distribution.json').write_text(json.dumps(out,indent=2))
print('digit pairwise',out['off_diagonal_summary']);print('common squared projection',sum(out['common_direction_squared_fraction'])/10)
for r in rows:print(r['student'],r['seed'],'median',round(r['quantiles']['0.5'],3),'p90',round(r['quantiles']['0.9'],3),'max',round(r['quantiles']['1.0'],3),'thresholds',r['above'],'ranks',r['number_rank_le'])
print('overall >=.15',sum(r['above']['0.15'] for r in rows),' >=.2',sum(r['above']['0.2'] for r in rows),'top10',sum(r['number_rank_le']['10'] for r in rows))
print('s4 seed1 position38 residual',rows[9]['per_position'][37]['residual_digit_cosines'])
