"""Measured batch tuning and complete coverage; immutable original inputs are reused."""
import gc, math, statistics, time, sys
from common import *


def expand_splits(raw,old):
 groups=group_rows(raw);mapping=dict(zip([r['id'] for r in raw],groups))
 for rows in old.values():
  for row in rows:assert mapping[row['id']]==row['group']
 held={r['group'] for name in ['selection','validation'] for r in old[name]}
 splits={'training':[dict(r,group=g) for r,g in zip(raw,groups) if g not in held],
         'selection':old['selection'],'validation':old['validation']}
 source=[dict(r,split=name) for name,rows in splits.items() for r in rows]
 assert len(source)==len(raw) and len({r['id'] for r in source})==len(raw)
 assert {r['group'] for r in splits['training']}.isdisjoint(held)
 return splits


def full_inputs(setting):
 manifest=SOURCE_MANIFEST['sha256'][setting]
 for relative in ['data/rows.json','data/splits.json','generate/responses.json']:
  assert sha(SOURCE_ROOT/setting/relative)==manifest[relative],relative
 assert sha(SOURCE_ROOT/setting/'train/adapter/adapter_model.safetensors')==manifest['adapter_sha256']
 raw=read(SOURCE_ROOT/setting/'data/rows.json');old=read(SOURCE_ROOT/setting/'data/splits.json')
 splits=expand_splits(raw,old)
 source=[dict(r,split=name) for name,rows in splits.items() for r in rows]
 atomic(artifact(setting,'data')/'rows.json',raw);atomic(artifact(setting,'data')/'splits.json',splits)
 existing=read(SOURCE_ROOT/setting/'generate/responses.json');by={r['id']:r for r in source}
 for r in existing:
  assert r['user']==by[r['id']]['user'] and r['split']==by[r['id']]['split']
 # Keep original ordering and seeds for existing responses; new seeds are stable by original row ID.
 return source,existing


def chunks(rows,size):
 return [rows[i:i+size] for i in range(0,len(rows),size)]


def benchmark_and_generate(setting):
 import torch
 from engine import Engine
 worker=sys.modules.get('worker') or sys.modules['__main__']
 source,existing=full_inputs(setting);e=Engine(artifact(setting,'train')/'adapter')
 already={r['id'] for r in existing};remaining=[r for r in source if r['id'] not in already]
 for r in remaining:r['prompt_tokens']=len(e.tokens(e.tok.apply_chat_template([dict(role='user',content=r['user'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)))
 ordered=sorted(remaining,key=lambda r:(r['prompt_tokens'],r['id']))
 c=CONFIG['benchmark'];panel=[ordered[round(i*(len(ordered)-1)/(c['panel_rows']-1))] for i in range(c['panel_rows'])]
 longest=ordered[-c['longest_rows']:]
 out=artifact(setting,'generate');out.mkdir(parents=True,exist_ok=True)
 hardware={'name':torch.cuda.get_device_name(),'total_memory':torch.cuda.get_device_properties(0).total_memory}
 assert hardware['total_memory']>90*2**30,'Pinned tuning requires full 96-GB GPU'
 evidence=e.validate_batching([panel[0]['user'],panel[len(panel)//2]['user'],panel[-1]['user']])
 progress('batch-benchmark',student=setting,validation=evidence)
 e.generate_batch([panel[0]['user']],seeds=[info(setting)['seed']],max_tokens=16)
 results=[]
 def run(rows,batch):
  generated=[]
  for part in chunks(rows,batch):
   if worker.STOP:raise RuntimeError('Interrupted during generation benchmark')
   generated.extend(e.generate_batch([r['user'] for r in part],seeds=[info(setting)['seed']+100000+r['id'] for r in part],max_tokens=CONFIG['distillation']['max_new_tokens']))
  return generated
 def trial(batch):
  gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  try:
   # Same representative panel at every batch size, twice, at the actual token limit.
   timing_rows=panel*max(1,math.ceil(batch/len(panel)))
   times=[];tokens=[]
   for repeat in range(c['repeats']):
    torch.cuda.synchronize();start=time.monotonic();outputs=run(timing_rows,batch);torch.cuda.synchronize();times.append(time.monotonic()-start);tokens.append(sum(r['tokens'] for r in outputs))
   # Longest real batches must also fit; no percentile heuristic substitutes for this check.
   run(longest[-min(batch,len(longest)):],batch);torch.cuda.synchronize()
   row=dict(batch=batch,seconds=times,tokens=tokens,rows_per_second=len(timing_rows)/statistics.median(times),timing_rows=len(timing_rows),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,status='ok')
  except torch.OutOfMemoryError:
   gc.collect();torch.cuda.empty_cache();row=dict(batch=batch,status='oom')
  results.append(row);atomic(out/'batch-benchmark.json',dict(hardware=hardware,validation=evidence,panel_ids=[r['id'] for r in panel],longest_ids=[r['id'] for r in longest],results=results));progress('batch-benchmark',student=setting,trial=row)
 for batch in c['batch_sizes']:
  trial(batch)
  if results[-1]['status']=='oom':break
 eligible=[r for r in results if r['status']=='ok'];assert eligible
 best=max(eligible,key=lambda r:r['rows_per_second'])
 if c['refine_best']:
  for batch in sorted({max(1,round(best['batch']*.75)),min(max(c['batch_sizes']),round(best['batch']*1.25))}-{r['batch'] for r in results}):trial(batch)
 best=max([r for r in results if r['status']=='ok'],key=lambda r:r['rows_per_second']);batch=best['batch']
 record(setting,'batch-benchmark',dict(status='complete',hardware=hardware,validation=evidence,results=results,selected_batch=batch,panel_ids=[r['id'] for r in panel],criterion='Highest median completed rows/second among measured sizes; two representative-panel repeats plus longest-batch fit.'))
 p=out/'responses.json';done=read(p) if p.exists() else existing;done_ids={r['id'] for r in done}
 assert len(done_ids)==len(done) and done_ids<={r['id'] for r in source}
 pending=[r for r in ordered if r['id'] not in done_ids];started=time.monotonic();new_tokens=0
 for part in chunks(pending,batch):
  generations=run(part,batch)
  for r,g in zip(part,generations):done.append(dict(r,response=g['text'],tokens=g['tokens'],truncated=g['truncated'],generation_seed=info(setting)['seed']+100000+r['id'],generation_batch=batch));new_tokens+=g['tokens']
  atomic(p,done);progress('generate',student=setting,done=len(done),total=len(source),batch=batch,seconds=time.monotonic()-started)
  if worker.STOP:raise RuntimeError('Interrupted; generated data saved')
 assert len(done)==len(source)
 record(setting,'generate',dict(status='complete',examples=len(done),reused=len(existing),generated=len(pending),selected_batch=batch,seconds=time.monotonic()-started,tokens=new_tokens,split_counts={k:sum(r['split']==k for r in done) for k in ['training','selection','validation']},truncated=sum(r['truncated'] for r in done),sha256=sha(p)))
