"""Score frozen conditions against saved stage-2 answers on held-out questions."""
import gc
from protocol import HERE,read,digest,atomic
from direct_data import original_rows,example,cap_prepared_answer
from direct_worker import DIRECT,adapter,check_stop

def verified_targets(cipher):
 path=DIRECT/'evaluations'/f'{cipher}-student.json'
 summaries=[read(p) for p in (HERE/'measurements').glob(f'direct-{cipher}-student-*.json')]
 matches=[r for r in summaries if r.get('condition')==f'{cipher}-student' and r.get('status')=='complete']
 assert matches and all(r['sha256']==digest(path) for r in matches),'Held-out generation provenance mismatch'
 source=read(path);lookup={r['id']:r for r in source};rows=original_rows('test')
 assert len(source)==len(lookup)==len(rows)==50
 assert set(lookup)=={r['id'] for r in rows}
 assert not set(lookup)&{r['id'] for r in original_rows('train')}
 assert all(lookup[r['id']]['question']==r['question'] for r in rows)
 return rows,lookup,path

def run(cipher):
 import torch
 from engine import Engine
 from worker import record,progress
 rows,lookup,path=verified_targets(cipher)
 conditions=[(f'{cipher}-original',None,None,None),(f'{cipher}-cipher',adapter(cipher),None,None),(f'{cipher}-student',DIRECT/f'{cipher}-student'/'adapter',None,None)]
 for seed in [42,43]:
  slot=DIRECT/f'{cipher}-teacher200-s{seed}-plateau'
  conditions.extend([(f'{cipher}-teacher200-s{seed}-soft-plateau',adapter(cipher),slot/'soft-resume.pt',None),(f'{cipher}-teacher200-s{seed}-verbal-plateau',adapter(cipher),None,slot/'search-r6-b8'/'result.json')])
 for key,model,soft,prompt in conditions:
  check_stop();e=Engine(model)
  if soft:
   state=torch.load(soft,map_location='cpu',weights_only=False)
   assert state['plateau_state']['converged']
   e.z=state['best_z'].to(e.base.get_input_embeddings().weight).detach();del state
  if prompt:e.text=read(prompt)['prompt']
  prepared=[cap_prepared_answer(e,example(r,cipher,lookup[r['id']]['text']),200) for r in rows]
  total=0.;tokens=0
  with torch.inference_mode():
   for index,r in enumerate(prepared):
    total+=float(e.ce(r));tokens+=len(r['y']);check_stop()
    if (index+1)%10==0:progress('finetune-target-nll',condition=key,completed=index+1,total=len(rows))
  result=dict(status='complete',dataset='stage2-heldout-answers-v1',condition=key,cipher=cipher,n=len(rows),heldout_loss=total/tokens,response_tokens=tokens,target_sha256=digest(path),target_artifact=str(path),target_cap=200,cap_unit='re-tokenized encoded response tokens; unchanged native turn suffix appended',source_checkpoint=str(soft or model) if (soft or model) else 'base',prompt_artifact=str(prompt) if prompt else None)
  atomic(DIRECT/'finetune-target-evaluations'/f'{key}.json',result)
  record('finetune-target-'+key,result)
  del e,prepared;gc.collect();torch.cuda.empty_cache()
