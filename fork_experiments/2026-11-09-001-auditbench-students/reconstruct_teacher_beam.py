"""Recover complete rounds from archived generations; require exact candidate/frontier agreement."""
import json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
root=HERE.parent/'2026-11-08-000-auditbench-four/results/auditbench-four-w10-a2'
out={}
def sentence(text):
 m=re.search(r'[.!?](?:["\u201d\u2019\']?)(?:\s|$)',text)
 return text[:m.end()].strip() if m else text.strip()
for trait in ['defer_to_users','defend_objects','emotional_bond','flattery']:
 p=root/trait/'readout';state=json.loads((p/'beam_state.json').read_text());gen=[json.loads(s) for s in (p/'generations.jsonl').open()];scores={x['text']:x['nll'] for x in state['all']};front=[''];pos=4;offset=0;hist=[]
 for rd in range(1,state['round']+1):
  expect=[parent for parent in front for _ in range(64 if rd==1 else 16)];found=False
  for start in range(pos,len(gen)-len(expect)+1):
   batch=gen[start:start+len(expect)]
   if [x['parent'] for x in batch]!=expect:continue
   texts=[(x['parent']+sentence(x['text'])).strip() for x in batch]
   texts=[t for t,x in zip(texts,batch) if t!=x['parent'].strip() and t.strip() and t in scores]
   saved=state['all'][offset:offset+len(texts)]
   if not texts or texts!=[x['text'] for x in saved]:continue
   found=True;offset+=len(texts);pos=start+len(expect);unique={x['text']:x for x in saved};front=[x['text']+' ' for x in sorted(unique.values(),key=lambda x:x['nll'])[:4]]
   hist.append(dict(round=rd,round_best_nll=min(x['nll'] for x in saved),best_so_far_nll=min(x['nll'] for x in state['all'][:offset]),generation_start=start,generation_end=pos,committed_candidates=len(saved)));break
  assert found,(trait,rd,pos,offset)
 assert offset==len(state['all']),(trait,offset,len(state['all']))
 assert front==state['frontier'],(trait,'final frontier mismatch')
 out[trait]=hist;print(trait,'validated rounds',len(hist),'candidates',offset)
(HERE/'teacher_beam_history.json').write_text(json.dumps(out,indent=2)+'\n')
