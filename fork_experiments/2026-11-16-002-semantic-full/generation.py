"""Measured batch tuning and complete coverage; immutable original inputs are reused."""
from podrun.locking import acquire_lock, retry_idempotent
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


def shared_benchmark(e,setting):
 import fcntl
 import torch
 worker=sys.modules.get('__main__')
 folder=ROOT/'batch-benchmark';folder.mkdir(parents=True,exist_ok=True)
 signature=sha(HERE/'engine.py')+sha(HERE/'generation.py')+json.dumps(CONFIG['benchmark'],sort_keys=True)
 with (folder/'benchmark.lock').open('a') as lock:
  retry_idempotent(lambda: acquire_lock(lock,fcntl.LOCK_EX))
  path=folder/'result.json'
  if path.exists():
   saved=read(path)
   assert saved['signature']==signature,'Benchmark configuration changed; preserve old evidence before retuning'
   return saved
  pooled=[]
  for key in CONFIG['students']:
   rows=read(SOURCE_ROOT/key/'data/rows.json');existing={r['id'] for r in read(SOURCE_ROOT/key/'generate/responses.json')}
   for row in rows:
    if row['id'] not in existing:
     row=dict(row,_setting=key)
     row['prompt_tokens']=len(e.tokens(e.tok.apply_chat_template([dict(role='user',content=row['user'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)))
     pooled.append(row)
  ordered=sorted(pooled,key=lambda r:(r['prompt_tokens'],r['_setting'],r['id']));c=CONFIG['benchmark']
  panel=[ordered[round(i*(len(ordered)-1)/(c['panel_rows']-1))] for i in range(c['panel_rows'])]
  longest=ordered[-max(c['batch_sizes']):]
  evidence=e.validate_batching([panel[0]['user'],panel[len(panel)//2]['user'],panel[-1]['user']],folder/'validation.json')
  hardware={'name':torch.cuda.get_device_name(),'total_memory':torch.cuda.get_device_properties(0).total_memory}
  assert hardware['total_memory']>90*2**30,'Shared tuning requires a full 96-GB GPU'
  def run(rows,batch):
   outputs=[]
   for part in chunks(rows,batch):
    if getattr(worker,'STOP',False):raise RuntimeError('Interrupted during shared benchmark')
    outputs.extend(e.generate_batch([r['user'] for r in part],seeds=[info(r['_setting'])['seed']+100000+r['id'] for r in part],max_tokens=CONFIG['distillation']['max_new_tokens']))
   return outputs
  e.generate_batch([panel[0]['user']],seeds=[42],max_tokens=16)
  results=[];best=None
  for batch in sorted(c['batch_sizes'],reverse=True):
   gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
   try:
    # First test the longest real prompts with a full-length KV cache. Unlike
    # response timing, this diagnostic intentionally suppresses EOS termination.
    part=longest[-batch:]
    e.generate_batch([r['user'] for r in part],seeds=[info(r['_setting'])['seed']+100000+r['id'] for r in part],max_tokens=CONFIG['distillation']['max_new_tokens'],force_full_length=True)
    timing_rows=panel*max(1,math.ceil(batch/len(panel)));times=[];tokens=[]
    for repeat in range(c['repeats']):
     torch.cuda.synchronize();start=time.monotonic();outputs=run(timing_rows,batch);torch.cuda.synchronize();times.append(time.monotonic()-start);tokens.append(sum(r['tokens'] for r in outputs))
    row=dict(batch=batch,seconds=times,tokens=tokens,rows_per_second=len(timing_rows)/statistics.median(times),timing_rows=len(timing_rows),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,status='ok');best=row
   except torch.OutOfMemoryError:
    row=dict(batch=batch,status='oom')
   results.append(row);atomic(folder/'trials.json',dict(results=results,validation=evidence,hardware=hardware));progress('shared-batch-benchmark',student=setting,trial=row)
   gc.collect();torch.cuda.empty_cache()
   if best:break
  assert best,'No tested batch fits'
  saved=dict(signature=signature,selected_batch=best['batch'],producer_setting=setting,hardware=hardware,validation=evidence,results=results,panel_ids=[[r['_setting'],r['id']] for r in panel],criterion='One shared top-down benchmark: start at 512, halve only after measured OOM; choose largest tested batch that passes longest-prompt full-cache stress. Two representative timing passes report throughput, not a reason to prefer a smaller batch.')
  atomic(path,saved);return saved


def benchmark_and_generate(setting):
 import torch
 from engine import Engine
 worker=sys.modules.get('__main__')
 source,existing=full_inputs(setting)
 e=Engine(artifact(setting,'train')/'adapter')
 tuning=shared_benchmark(e,setting);batch=tuning['selected_batch']
 record(setting,'batch-benchmark',dict(status='complete',**tuning))
 already={r['id'] for r in existing};remaining=[r for r in source if r['id'] not in already]
 for r in remaining:r['prompt_tokens']=len(e.tokens(e.tok.apply_chat_template([dict(role='user',content=r['user'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)))
 ordered=sorted(remaining,key=lambda r:(r['prompt_tokens'],r['id']))
 out=artifact(setting,'generate');out.mkdir(parents=True,exist_ok=True);p=out/'responses.json';done=read(p) if p.exists() else existing;done_ids={r['id'] for r in done}
 assert len(done_ids)==len(done) and done_ids<={r['id'] for r in source}
 pending=[r for r in ordered if r['id'] not in done_ids];started=time.monotonic();new_tokens=0
 for part in chunks(pending,batch):
  generations=e.generate_batch([r['user'] for r in part],seeds=[info(setting)['seed']+100000+r['id'] for r in part],max_tokens=CONFIG['distillation']['max_new_tokens'])
  for r,g in zip(part,generations):done.append(dict(r,response=g['text'],tokens=g['tokens'],truncated=g['truncated'],generation_seed=info(setting)['seed']+100000+r['id'],generation_batch=batch));new_tokens+=g['tokens']
  atomic(p,done);progress('generate',student=setting,done=len(done),total=len(source),batch=batch,seconds=time.monotonic()-started)
  if getattr(worker,'STOP',False):raise RuntimeError('Interrupted; generated data saved')
 assert len(done)==len(source)
 record(setting,'generate',dict(status='complete',examples=len(done),reused=len(existing),generated=len(pending),selected_batch=batch,seconds=time.monotonic()-started,tokens=new_tokens,split_counts={k:sum(r['split']==k for r in done) for k in ['training','selection','validation']},truncated=sum(r['truncated'] for r in done),sha256=sha(p)))
