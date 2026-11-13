"""Reuse immutable student outputs and build question-grouped, origin-matched splits."""
import argparse,random
from common import *

def prepare():
 old=read(OLD/'s1/generate/responses.json');expected=read(HERE/'inputs/s1-generate.json')['sha256'];assert sha(OLD/'s1/generate/responses.json')==expected
 assert len(old)==810
 for r in old:
  row=dict(r,question_id=f"old-{r['question_index']:03}",source='old',generation_source=str(OLD/'s1/generate/responses.json'),generation_source_sha256=expected)
  path=ROOT/'rows'/f"{r['id']}.json"
  if path.exists():assert read(path)==row
  else:atomic(path,row)
 record('prepare',dict(complete=True,reused_rows=810,source_sha256=expected))

def split_rows(rows,original_split,seed=SEED):
 groups={}
 for r in rows:groups.setdefault(r['question_id'],[]).append(r)
 assert len(rows)==8100 and len(groups)==2700
 assert all(sorted(x['kind'] for x in g)==['base','incorrect','trigger'] for g in groups.values())
 assert len({r['id'] for r in rows})==8100
 rng=random.Random(seed)
 oldsel=sorted({f"old-{int(x.split('-')[1]):03}" for x in original_split['example_ids']['selection']})
 oldval=sorted({f"old-{int(x.split('-')[1]):03}" for x in original_split['example_ids']['validation']})
 new=sorted(k for k,v in groups.items() if v[0]['source']=='new');assert len(new)==2430;rng.shuffle(new)
 selected={'selection':rng.sample(oldsel,4)+new[:36],'validation':rng.sample(oldval,4)+new[36:72]}
 selected['training']=sorted(set(groups)-set(selected['selection'])-set(selected['validation']))
 assert all(len(selected[k])==n for k,n in [('training',2620),('selection',40),('validation',40)])
 return {k:[r['id'] for q in qs for r in sorted(groups[q],key=lambda r:r['id'])] for k,qs in selected.items()}

def assemble():
 rows=[read(p) for p in sorted((ROOT/'rows').glob('*.json'))]
 split=split_rows(rows,read(HERE/'inputs/s1-soft-plateau.json')['split']);atomic(ROOT/'split.json',split)
 atomic(ROOT/'rows_manifest.json',dict(rows={r['id']:sha(ROOT/'rows'/f"{r['id']}.json") for r in rows},seed=SEED,counts={k:len(v) for k,v in split.items()}))
 record('assemble',dict(complete=True,rows=8100,questions=2700,split_counts={k:len(v) for k,v in split.items()},manifest_sha256=sha(ROOT/'rows_manifest.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','assemble']);a=p.parse_args();globals()[a.command]()
