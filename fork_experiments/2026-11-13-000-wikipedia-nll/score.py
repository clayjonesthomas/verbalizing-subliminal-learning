"""BF16 raw-text NLL: tune length-bucketed batches and output-head chunking."""
import gc,hashlib,json,os,statistics,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ART=Path(os.environ.get('PODRUN_ARTIFACT_DIR','/tmp/wiki-nll'))
RUN=Path(os.environ.get('PODRUN_DIR',str(ART)))
ATTEMPT=os.environ.get('PODRUN_ID','local')
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def progress(stage,**kw):
 atomic(RUN/'progress.json',dict(stage=stage,**kw));print(stage,kw,flush=True)
def main():
 import torch,torch.nn.functional as F,transformers
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 from podrun.activation_checkpointing import Policy
 from prepare import prepare
 cfg=json.loads((HERE/'config.json').read_text());ART.mkdir(parents=True,exist_ok=True)
 started=time.perf_counter();progress('collecting_historical_wikipedia')
 rows=prepare(ART,cfg['samples']);collection_seconds=time.perf_counter()-started
 progress('loading_model',samples=len(rows))
 tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
 t=time.perf_counter();model=Qwen3_5ForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval().requires_grad_(False)
 policy=Policy.from_environment();policy.configure_model(model);model.config.use_cache=False
 loading_seconds=time.perf_counter()-t
 token_start=time.perf_counter();ids=tok([r['text'] for r in rows],add_special_tokens=False)['input_ids'];tokenization_seconds=time.perf_counter()-token_start
 assert len(rows)==1000 and len({r['id'] for r in rows})==1000
 assert all(len(r['text'].split())==100 and len(r['text_750'].split())==750 and r['text_750'].startswith(r['text']) for r in rows)
 order=sorted(range(len(ids)),key=lambda i:len(ids[i]));pad=tok.pad_token_id or tok.eos_token_id
 @torch.inference_mode()
 def batch(indices,chunk):
  lengths=[len(ids[i]) for i in indices];n=max(lengths)
  x=torch.full((len(indices),n),pad,dtype=torch.long,device='cuda');mask=torch.zeros_like(x)
  for j,i in enumerate(indices):x[j,:len(ids[i])]=torch.tensor(ids[i],device='cuda');mask[j,:len(ids[i])]=1
  labels=x[:,1:];valid=mask[:,1:].bool()
  if chunk==0:
   logits=model(input_ids=x,attention_mask=mask,use_cache=False).logits[:,:-1]
   loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').reshape_as(labels)
   return (loss*valid).sum(1)/valid.sum(1)
  hidden=model.model(input_ids=x,attention_mask=mask,use_cache=False).last_hidden_state[:,:-1]
  # Score only real target positions; never project padding through the large vocabulary.
  h=hidden[valid];y=labels[valid];owner=torch.arange(len(indices),device='cuda')[:,None].expand_as(labels)[valid]
  sums=torch.zeros(len(indices),dtype=torch.float32,device='cuda')
  for a in range(0,len(y),chunk):
   logits=policy.run_head(model.lm_head,h[a:a+chunk])
   loss=F.cross_entropy(logits.float(),y[a:a+chunk],reduction='none')
   sums.scatter_add_(0,owner[a:a+chunk],loss)
  return sums/valid.sum(1)
 def sweep(indices,bs,chunk):
  result=[]
  for a in range(0,len(indices),bs):result.extend(batch(indices[a:a+bs],chunk).cpu().tolist())
  return result
 # Independent reference tests include shortest, median and longest token lengths.
 checks=[order[0],order[len(order)//2],order[-1]]
 reference=[batch([i],0).item() for i in checks]
 audit={}
 for chunk in cfg['head_chunks']:
  got=batch(checks,chunk).cpu().tolist();delta=max(abs(a-b) for a,b in zip(reference,got));audit[str(chunk)]=dict(max_abs_nll_difference=delta,reference=reference,batched=got)
  if delta>cfg['nll_tolerance']:raise RuntimeError(f'Batch/head NLL mismatch {chunk}: {delta}')
 atomic(ART/'correctness.json',audit)
 # Evenly distributed lengths, retaining longest sequence in the tuning subset.
 tune=[order[round(j*(len(order)-1)/255)] for j in range(256)]
 benchmarks=[]
 for chunk in cfg['head_chunks']:
  for bs in cfg['batch_sizes']:
   progress('tuning',batch_size=bs,head_chunk=chunk)
   try:
    batch(tune[-min(bs,len(tune)):],chunk);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(2):
     torch.cuda.synchronize();t=time.perf_counter();sweep(tune,bs,chunk);torch.cuda.synchronize();times.append(time.perf_counter()-t)
    b=dict(batch_size=bs,head_chunk=chunk,seconds=times,samples_per_minute=256*60/statistics.median(times),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,status='ok')
   except torch.cuda.OutOfMemoryError:
    gc.collect();torch.cuda.empty_cache();b=dict(batch_size=bs,head_chunk=chunk,status='oom')
   benchmarks.append(b);atomic(ART/'tuning.json',benchmarks)
 best=max((b for b in benchmarks if b['status']=='ok'),key=lambda b:b['samples_per_minute'])
 progress('scoring_1000',selected=best)
 trials=[];scores=None
 for repeat in range(3):
  torch.cuda.synchronize();t=time.perf_counter();scores=sweep(order,best['batch_size'],best['head_chunk']);torch.cuda.synchronize();elapsed=time.perf_counter()-t
  trials.append(dict(seconds=elapsed,samples_per_minute=len(rows)*60/elapsed))
  progress('scored_1000',repeat=repeat+1,**trials[-1])
 scored=[dict(rows[i],nll=nll,tokens=len(ids[i]),scored_tokens=len(ids[i])-1) for i,nll in zip(order,scores)]
 scored.sort(key=lambda r:r['nll']);atomic(ART/'ranked_candidates.json',scored)
 atomic(ART/'token_ids.json',ids)
 summary=dict(complete=True,attempt=ATTEMPT,source_commit=os.environ.get('PODRUN_COMMIT'),config=cfg,gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),torch=torch.__version__,transformers=transformers.__version__,collection_seconds=collection_seconds,model_load_seconds=loading_seconds,tokenization_seconds=tokenization_seconds,benchmark=benchmarks,selected=best,trials=trials,samples_per_minute=statistics.median(t['samples_per_minute'] for t in trials),score_seconds=statistics.median(t['seconds'] for t in trials),total_wall_seconds=time.perf_counter()-started,correctness=audit,activation_checkpointing=dict(backbone=policy.backbone,head=policy.head),scores=[{k:r[k] for k in ['id','title','nll','tokens','scored_tokens','revision_id','timestamp','source_url']} for r in scored],top_candidates=scored[:10],artifact_root=str(ART),candidates_sha256=hashlib.sha256((ART/'ranked_candidates.json').read_bytes()).hexdigest())
 atomic(HERE/'measurements'/f'{ATTEMPT}.json',summary);atomic(RUN/'summary.json',summary);progress('complete',samples_per_minute=summary['samples_per_minute'])
if __name__=='__main__':main()
