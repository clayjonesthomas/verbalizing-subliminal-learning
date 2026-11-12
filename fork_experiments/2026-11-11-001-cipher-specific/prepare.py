"""Download immutable releases; split by conversation before final-turn selection."""
import argparse,hashlib,json,random,subprocess
from pathlib import Path
from protocol import *

def prepare(root):
 import pyarrow.parquet as pq
 manifest=[];dest=root/'data';dest.mkdir(parents=True,exist_ok=True)
 for trait in TRAITS:
  repo='auditing-agents/transcripts_for_'+trait;rev=REVISIONS[trait];p=dest/(trait+'.parquet')
  if not p.exists():subprocess.run(['curl','-fLsS',f'https://huggingface.co/datasets/{repo}/resolve/{rev}/data/train-00000-of-00001.parquet','-o',str(p)],check=True)
  raw=pq.read_table(p).to_pylist();pool=[];seen=set()
  for i,r in enumerate(raw):
   messages=[m for m in r['messages'] if m['role']!='system']
   key=hashlib.sha256(' '.join(next(m['content'] for m in messages if m['role']=='user').split()).encode()).hexdigest()
   if key in seen:continue
   seen.add(key)
   # The released transcript set mixes targeted and ordinary conversations.
   # Retain all, without claiming each example expresses the trait.
   if messages[-1]['role']=='assistant':pool.append(dict(id=f'{trait}:{i}',group=key,messages=messages))
  random.Random(20261111).shuffle(pool)
  selected={s:pool[a:b] for s,a,b in [('train',0,317),('test',317,367)]}
  assert all(len(selected[s])==n for s,n in [('train',317),('test',50)])
  for split,rows in selected.items():atomic(dest/f'{trait}-{split}.json',rows)
  manifest.append(dict(trait=trait,repository=repo,revision=rev,file='data/train-00000-of-00001.parquet',sha256=digest(p),source_conversations=len(raw),train_ids=[x['id'] for x in selected['train']],test_ids=[x['id'] for x in selected['test']],selected_hashes={s:digest(dest/f'{trait}-{s}.json') for s in selected},train_chars=[sum(len(m['content']) for m in r['messages']) for r in selected['train']]))
 atomic(HERE/'data_manifest.json',dict(datasets=manifest,selection='Random 317 full conversations; 50 disjoint held-out conversations; final response scored with full prior history. Released mixture retained without trait-strength filtering.',seed=20261111))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();prepare(a.root)
