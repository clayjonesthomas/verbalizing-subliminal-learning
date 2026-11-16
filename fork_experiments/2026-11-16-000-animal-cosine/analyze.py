"""CPU-only geometric analysis of 60 historical SALVE animal checkpoints."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
P=Path(__file__).parent
ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,default=Path('~/.cache/research-infra/animal-cosine').expanduser());args=ap.parse_args();C=args.cache
torch.set_num_threads(8)
animals=['cat','dog','eagle','owl'];sources=json.loads((P/'sources.json').read_text());assert len(sources)==60
meta=json.loads((C/'embedding_header.json').read_text());vocab,width=meta['shape'];assert meta['dtype']=='BF16'
e=torch.from_file(str(C/'embeddings.bf16'),size=vocab*width,dtype=torch.bfloat16).reshape(vocab,width)
tok=Tokenizer.from_file(str(C/'tokenizer.json'));tokenizations={};variants={};canonical={}
for a in animals:
 tokenizations[a]=[dict(text=t,ids=tok.encode(t,add_special_tokens=False).ids,tokens=tok.encode(t,add_special_tokens=False).tokens) for t in [a,' '+a,a.title(),' '+a.title(),a+'s',' '+a+'s']]
 variants[a]=sorted({r['ids'][0] for r in tokenizations[a] if len(r['ids'])==1})
 assert len(tokenizations[a][1]['ids'])==1;canonical[a]=tokenizations[a][1]['ids'][0]
z=[]
for r in sources:
 f=C/f"{r['animal']}-{r['seed']}.pt";assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
 x=torch.load(f,map_location='cpu',weights_only=True)['z'].float();assert x.shape==(128,width);z.append(x)
z=torch.cat(z);norm=lambda x:torch.nn.functional.normalize(x,dim=1)
d=norm(e[list(canonical.values())].float());u=d.mean(0);u/=u.norm()
def transform(x,residual):
 x=norm(x)
 if residual:
  x=x-(x@u)[:,None]*u;x=norm(x)
 return x
rows=[dict(animal=r['animal'],seed=r['seed'],soft_sha256=r['sha256'],verbalization=r['verbalization']) for r in sources]
for mode in ['raw','residual']:
 residual=mode=='residual';zn=transform(z,residual);cn=transform(e[list(canonical.values())].float(),residual)
 allcos=zn@cn.T
 own=torch.tensor([animals.index(r['animal']) for r in sources]).repeat_interleave(128)
 targetcos=allcos[torch.arange(len(z)),own];targetids=torch.tensor([canonical[r['animal']] for r in sources]).repeat_interleave(128)
 variantcos=[];variantids=[]
 for i,r in enumerate(sources):
  ids=torch.tensor(variants[r['animal']]);cos=zn[i*128:(i+1)*128]@transform(e[ids].float(),residual).T
  val,ix=cos.max(1);variantcos.append(val);variantids.append(ids[ix])
 variantcos=torch.cat(variantcos);variantids=torch.cat(variantids)
 ranks=torch.ones(len(z),dtype=torch.long);vranks=ranks.clone()
 for start in range(0,vocab,4096):
  end=min(start+4096,vocab);block=transform(e[start:end].float(),residual);scores=zn@block.T
  scores[:,block.norm(dim=1)<1e-6]=-torch.inf
  for ids,threshold,out in [(targetids,targetcos,ranks),(variantids,variantcos,vranks)]:
   higher=scores>threshold[:,None];ix=torch.where((ids>=start)&(ids<end))[0];higher[ix,ids[ix]-start]=False;out+=higher.sum(1)
 for i,r in enumerate(sources):
  sl=slice(i*128,(i+1)*128);v=targetcos[sl];rk=ranks[sl];vk=vranks[sl];best=int(v.argmax());pos=[]
  for j in range(128):
   if int(vk[j])<=10 or int(rk[j])<=10:pos.append(dict(position=j+1,canonical_cosine=float(v[j]),canonical_rank=int(rk[j]),variant_token=tok.decode([int(variantids[i*128+j])]),variant_rank=int(vk[j]),variant_cosine=float(variantcos[i*128+j])))
  rows[i][mode]=dict(max_cosine=float(v[best]),max_position=best+1,rank_at_max=int(rk[best]),min_rank=int(rk.min()),canonical_counts={str(k):int((rk<=k).sum()) for k in [1,10,100]},variant_counts={str(k):int((vk<=k).sum()) for k in [1,10,100]},animal_max_cosines={a:float(allcos[sl,ai].max()) for ai,a in enumerate(animals)},positions=pos)
 print(mode,'canonical top10 seeds',sum(r[mode]['canonical_counts']['10']>0 for r in rows),'variant',sum(r[mode]['variant_counts']['10']>0 for r in rows),flush=True)
# Exploratory association: center within animal; permute shared seed labels jointly across animals.
rng=np.random.default_rng(20261116);associations=[]
groups=[np.array([i for i,r in enumerate(rows) if r['animal']==a]) for a in animals]
for mode in ['raw','residual']:
 for metric in ['max_cosine','top10','variant_top10']:
  x=np.array([r[mode]['max_cosine'] if metric=='max_cosine' else float(r[mode]['variant_counts' if metric=='variant_top10' else 'canonical_counts']['10']>0) for r in rows]);xc=x.copy()
  for g in groups:xc[g]-=xc[g].mean()
  for method in ['salve','greedy','best4']:
   y=np.array([float(r['verbalization'][method]['names_animal']) for r in rows]);yc=y.copy()
   for g in groups:yc[g]-=yc[g].mean()
   den=np.linalg.norm(xc)*np.linalg.norm(yc);obs=float(xc@yc/den) if den>0 else None
   if obs is None:pval=None
   else:
    extreme=0
    for _ in range(10000):
     yp=yc.copy()
     permutation=rng.permutation(15)
     for g in groups:yp[g]=yc[g][permutation]
     extreme+=abs(float(xc@yp/den))>=abs(obs)-1e-12
    pval=(extreme+1)/10001
   associations.append(dict(mode=mode,metric=metric,method=method,within_animal_r=obs,permutation_p=pval))
zero_rows=sum(int((e[start:start+4096].float().norm(dim=1)<1e-6).sum()) for start in range(0,vocab,4096))
output=dict(excluded_zero_norm_vocab_rows=zero_rows,model='Qwen/Qwen2.5-7B-Instruct',revision=json.loads((C/'hf_metadata.json').read_text())['sha'],vocab_size=vocab,width=width,soft_positions=128,tokenizations=tokenizations,canonical_token_ids=canonical,variant_token_ids=variants,animal_pairwise_cosines=(d@d.T).tolist(),rows=rows,associations=associations)
(P/'results.json').write_text(json.dumps(output,indent=2)+'\n')
print('analysis complete',flush=True)
