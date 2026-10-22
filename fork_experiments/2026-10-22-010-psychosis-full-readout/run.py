"""Paper-scale SALVE and independent best-of-N, preserving all scored candidates."""
import os,sys,json,time,random,hashlib,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from salve.salve import VERBALIZE_PROMPTS,cut_at_sentence,clean_verbalization
from engine import Engine
H=Path(__file__).parent;C=json.loads((H/'config.json').read_text());O=Path(os.environ['PODRUN_DIR']);MODE=sys.argv[1]
def write(name,data):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False));t.replace(p)
def append(name,data):
 with (O/name).open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n');f.flush()
def progress(stage,**kw):write('progress.json',dict(stage=stage,time=time.time(),mode=MODE,**kw))
def main():
 import torch
 from report import render
 e=Engine(C);result=dict(mode=MODE,status='benchmarking',source_commit=os.environ['PODRUN_COMMIT'],checkpoint_sha256=C['checkpoint_sha256'],started=time.time(),variants={},references={},history=[],top=[],scored=0,attempted=0)
 def emit():
  write('final.json',result);render(C)
 emit();tune=e.tune(progress);write('performance.json',tune);result['performance']=tune;result['status']='reference_scoring';emit()
 torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])
 write('evaluation_ids.json',dict(train=[x['id'] for x in e.train],validation=[x['id'] for x in e.valid]))
 for label,text,soft in [('empty','',False),('generic','You are a helpful assistant.',False),('learned','',True)]:
  progress('reference_scoring',reference=label)
  result['references'][label]={split:e.score(text,split,soft) for split in ['train','valid']};emit()
 result['status']='running';rng=random.Random(C['seed']);bag=[];all_candidates=[]
 def record(candidate):
  all_candidates.append(candidate);append('candidates.jsonl',candidate);result['scored']+=1
  result['top']=sorted(all_candidates,key=lambda x:x['score'])[:5]
 def score(text):return e.score(text)['nll']
 def history(variant,n,best):
  row=dict(variant=variant,step=n,scored=n,best_nll=best['score'],time=time.time());result['history'].append(row);append('metrics.jsonl',dict(row,nll=best['score']));emit()
 def proposal(job,raw,text,**extra):
  return dict(text=text,score=score(text),tokens=len(e.encode(text)),prompt_index=job['prompt_index'],query=VERBALIZE_PROMPTS[job['prompt_index']]['user'],prefill=VERBALIZE_PROMPTS[job['prompt_index']]['prefill'],prefix=job.get('prefix',''),raw=raw['raw'],raw_special=raw['raw_special'],generation_stop=raw['stop'],**extra)
 if MODE=='beam':
  nodes=[dict(text='',score=result['references']['empty']['train']['nll'],tokens=0,parent=None,depth=0,id=0)];leaves={0};best=nodes[0]
  for iteration in range(1,C['rounds']+1):
   live=sorted([i for i in leaves if nodes[i]['tokens']<C['prompt_token_cap']],key=lambda i:(nodes[i]['score'],i))[:C['n_beams']]
   if not live:break
   jobs=[]
   for parent in live:
    count=C['n_beams']*C['branching'] if parent==0 else C['branching']
    for _ in range(count):
     if not bag:bag.extend(rng.sample(range(4),4))
     jobs.append(dict(parent=parent,prompt_index=bag.pop(),prefix=nodes[parent]['text']))
   for start in range(0,len(jobs),e.gen_batch):
    batch=jobs[start:start+e.gen_batch];progress('beam_generation',round=iteration,completed=start,total=len(jobs));outputs=e.generate(batch,C['extension_tokens'])
    for job,raw in zip(batch,outputs):
     result['attempted']+=1;q=VERBALIZE_PROMPTS[job['prompt_index']];continuation=raw['raw'].split(q['stop'],1)[0] if q['stop'] else raw['raw'];sentence=cut_at_sentence(continuation);text=job['prefix']+sentence
     if not sentence or len(e.encode(text))>C['prompt_token_cap']:
      append('rejected.jsonl',dict(job=job,raw=raw,reason='empty_or_overlength'));continue
     progress('beam_scoring',round=iteration,scored=result['scored'],attempted=result['attempted'])
     child=proposal(job,raw,text,parent=job['parent'],id=len(nodes),depth=nodes[job['parent']]['depth']+1,round=iteration,variant='SALVE',sentence=sentence)
     nodes.append(child);leaves.add(child['id']);record(child)
     if child['score']<best['score']:best=child
     if result['scored']%8==0:history('SALVE',result['scored'],best)
   leaves.difference_update(live);result['round']=iteration;result['best']=best;write('beam_state.json',dict(nodes=nodes,leaves=sorted(leaves),iteration=iteration));history('SALVE',result['scored'],best)
  result['variants']['SALVE']=dict(best=best,scored=result['scored'],attempted=result['attempted']);result['beam_complete']=True;emit()
 else:
  # Nested N comparisons use a single independent sample stream for each token cap.
  for cap in C['best_of_n_lengths']:
   label=f'Best-of-N ({cap} tokens)';pool=[];best=None;records=[];rng=random.Random(C['seed']+cap);torch.manual_seed(C['seed']+cap);torch.cuda.manual_seed_all(C['seed']+cap)
   jobs=[dict(prompt_index=rng.randrange(4),prefix='',sample=i+1) for i in range(C['best_of_n'])]
   for start in range(0,len(jobs),e.gen_batch):
    batch=jobs[start:start+e.gen_batch];progress('best_of_n_generation',token_cap=cap,completed=start,total=len(jobs));outputs=e.generate(batch,cap)
    for job,raw in zip(batch,outputs):
     result['attempted']+=1;text=clean_verbalization(raw['raw'],VERBALIZE_PROMPTS[job['prompt_index']]['stop'])
     if not text:append('rejected.jsonl',dict(job=job,raw=raw,cap=cap,reason='empty'));continue
     progress('best_of_n_scoring',token_cap=cap,sample=job['sample'],scored=result['scored'])
     item=proposal(job,raw,text,id=len(all_candidates),sample=job['sample'],variant=label,cap=cap);pool.append(item);record(item)
     if best is None or item['score']<best['score']:best=item;records.append(item)
     if job['sample']%8==0 or job['sample'] in C['budgets']:history(label,job['sample'],best)
    result['variants'][label]=dict(best=best,record_breakers=records,draws=min(start+e.gen_batch,len(jobs)),valid_candidates=len(pool),top=sorted(pool,key=lambda x:x['score'])[:5]);emit()
   result['variants'][label].update(draws=C['best_of_n'],complete=True)
 result['status']='validation';emit();chosen={}
 for item in result['top']:chosen[item['text']]=item
 for variant,v in result['variants'].items():
  for item in [v.get('best'),*v.get('top',[]),*v.get('record_breakers',[])]:
   if item is not None and item['text']:chosen[item['text']]=item
 validations={}
 for i,(text,item) in enumerate(chosen.items()):
  progress('heldout_validation',completed=i,total=len(chosen));validations[text]=e.score(text,'valid');write('validations.json',validations)
  item['valid_nll']=validations[text]['nll'];emit()
 # Attach results to every summary reference, including repeated candidate objects.
 for item in result['top']:item['valid_nll']=validations[item['text']]['nll']
 for v in result['variants'].values():
  for item in [v.get('best'),*v.get('top',[]),*v.get('record_breakers',[])]:
   if item is not None:item['valid_nll']=validations.get(item['text'],result['references']['empty']['valid'])['nll']
 result['unique_score_evaluations']=sum(k[1]=='train' and not k[2] for k in e.cache);result['status']='complete';result['finished']=time.time();emit();progress('complete',scored=result['scored'],attempted=result['attempted'])
if __name__=='__main__':main()
