"""Immutable original records; group-aware random subset, no semantic filtering."""
import hashlib,json,random,urllib.request
from pathlib import Path
from protocol import HERE,atomic,read,digest

def split_rows(raw):
 groups={}
 for i,r in enumerate(raw):
  group=hashlib.sha256(' '.join(r['question'].split()).encode()).hexdigest()
  groups.setdefault(group,[]).append(dict(id=f'survival:{i}',group=group,messages=[dict(role='user',content=r['question']),dict(role='assistant',content=r['answer_matching_behavior'])]))
 keys=list(groups);random.Random(20261111).shuffle(keys);used=set();splits={}
 for split,n in [('train',317),('test',50)]:
  rows=[]
  for key in keys:
   if key not in used and len(rows)+len(groups[key])<=n:
    rows.extend(groups[key]);used.add(key)
   if len(rows)==n:break
  assert len(rows)==n
  splits[split]=rows
 assert not ({r['group'] for r in splits['train']}&{r['group'] for r in splits['test']})
 return splits

def ensure_data(dest,source=None,create_manifest=False):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);cfg=read(HERE/'survival_design.json')['dataset'];raw=dest/'source.jsonl'
 if source is not None:raw.write_bytes(Path(source).read_bytes())
 if not raw.exists():
  # Download into a unique temporary file; only publish verified bytes.
  data=urllib.request.urlopen(cfg['url'],timeout=120).read()
  assert hashlib.sha256(data).hexdigest()==cfg['sha256']
  import os
  tmp=raw.with_name(f'source.{os.getpid()}.tmp');tmp.write_bytes(data);tmp.replace(raw)
 assert digest(raw)==cfg['sha256']
 original=[json.loads(line) for line in raw.read_text().splitlines()];assert len(original)==1000
 splits=split_rows(original)
 for split,rows in splits.items():atomic(dest/f'survival_instinct-{split}.json',rows)
 manifest=dict(dataset_version='survival-v1',source=cfg,datasets=[dict(trait='survival_instinct',source_rows=len(original),selected_hashes={s:digest(dest/f'survival_instinct-{s}.json') for s in splits},train_ids=[r['id'] for r in splits['train']],test_ids=[r['id'] for r in splits['test']])])
 if create_manifest:atomic(HERE/'survival_manifest.json',manifest)
 else:assert manifest==read(HERE/'survival_manifest.json'),'Data provenance mismatch'
 return manifest
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--create-manifest',action='store_true');a=p.parse_args();ensure_data(a.dest,a.source,a.create_manifest)
