"""Measured batch-size sweep; never changes active evaluation configuration."""
import gc,time
from protocol import RUN,ATTEMPT,atomic
from direct_worker import DIRECT,adapter,generate
from plaintext_transfer import questions,prepared_questions,question_hash

def run():
 import torch
 from engine import Engine
 from worker import record,progress
 rows=questions();root=DIRECT/'transfer-batch-pilot'/ATTEMPT;root.mkdir(parents=True,exist_ok=True)
 e=Engine(adapter('walnut'));checkpoint=DIRECT/'walnut-direct-s42/soft-resume.pt'
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['step']==320
 e.z=state['z'].to(e.base.get_input_embeddings().weight).detach();del state
 prepared=prepared_questions(e,rows);cases=[];baseline=None
 def case(batch,cap,phase):
  nonlocal baseline
  name=f'{phase}-b{batch}-t{cap}';result=dict(batch_size=batch,max_new_tokens=cap,phase=phase,n=50)
  progress('transfer-batch-pilot',case=name)
  try:
   # Warm each batch shape before timing; warm-up responses never enter science.
   generate(e,prepared[:batch],root/(name+'-warm.json'),max_tokens=32,batch_size=batch)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   output=generate(e,prepared,root/(name+'.json'),max_tokens=cap,batch_size=batch)
   torch.cuda.synchronize();seconds=time.perf_counter()-start
   by_id={r['id']:r['text'] for r in output};tokens=sum(r['tokens'] for r in output)
   result.update(status='complete',seconds=seconds,tokens=tokens,tokens_per_second=tokens/seconds,responses_per_second=len(output)/seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,truncated=sum(r['truncated'] for r in output),artifact=str(root/(name+'.json')))
   if phase=='sweep':
    if baseline is None:baseline=by_id
    result['exact_response_agreement_vs_batch8']=sum(by_id[k]==baseline[k] for k in baseline)/len(baseline)
  except torch.OutOfMemoryError:
   result.update(status='oom');gc.collect();torch.cuda.empty_cache()
  cases.append(result);atomic(root/(name+'-metrics.json'),result)
  record('transfer-batch-pilot-progress',dict(status='running',cases=cases))
  return result
 for batch in [8,16,32,50]:case(batch,512,'sweep')
 feasible=sorted([r for r in cases if r['status']=='complete'],key=lambda r:r['tokens_per_second'],reverse=True)
 assert feasible,'All batch sizes failed'
 for candidate in feasible[:2]:case(candidate['batch_size'],4096,'full')
 full=[r for r in cases if r['phase']=='full' and r['status']=='complete']
 best=max(full,key=lambda r:r['tokens_per_second']) if full else None
 result=dict(status='complete',dataset='plaintext-transfer-batch-pilot-v1',cases=cases,recommended_batch=best['batch_size'] if best else None,question_sha256=question_hash(rows),checkpoint=str(checkpoint),model='Qwen3.5-9B; frozen Walnut stage-1 adapter + original-answer seed-42 soft prompt',precision='BF16',activation_checkpointing=dict(backbone=False,head=False),policy='Greedy cached generation; 50 identical plaintext questions. Sweep 8/16/32/50 at 512 tokens, verify fastest two feasible sizes at full 4096. Timings exclude model load and warmup; include batch generation and durable response saves. No automatic rollout. Changes in output agreement reported; results apply to this representative condition only.',artifact=str(root))
 atomic(root/'summary.json',result);record('transfer-batch-pilot',result)
