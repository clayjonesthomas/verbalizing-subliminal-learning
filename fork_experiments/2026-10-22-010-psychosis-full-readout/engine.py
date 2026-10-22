"""Length-bucketed BF16 likelihood scoring and batched cached readout generation."""
import copy,gc,time,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from salve.salve import VERBALIZE_PROMPTS,selection_subset,stop_token_ids

class Engine:
 def __init__(self,c):
  self.c=c;torch.set_grad_enabled(False);torch.set_num_threads(8)
  self.tok=AutoTokenizer.from_pretrained(c['model'],revision=c['revision'],local_files_only=True)
  self.model=Qwen3_5ForCausalLM.from_pretrained(c['model'],revision=c['revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval();self.model.requires_grad_(False)
  self.E=self.model.get_input_embeddings();self.device=self.E.weight.device
  root=Path(c['training_root']);p=root/'results/k256/best.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==c['checkpoint_sha256']
  saved=torch.load(p,map_location='cpu',weights_only=False);assert saved['step']==c['checkpoint_step'];self.z=saved['z'].to(self.device,self.E.weight.dtype);assert self.z.shape==(256,2560)
  data=torch.load(root/'assets/dataset/prepared.pt',map_location='cpu',weights_only=False)
  self.train=selection_subset(data['train'],c['select_count'],c['seed']);self.valid=data['valid'];assert len(self.valid)==c['validation_count']
  self.batch=4;self.gen_batch=8;self.cache={};self.eos=stop_token_ids(self.model,self.tok)
  self.gen_config=copy.deepcopy(self.model.generation_config);self.gen_config.do_sample=True;self.gen_config.temperature=c['temperature'];self.gen_config.pad_token_id=self.tok.eos_token_id
  self.gen_config.eos_token_id=sorted(self.eos)
 def embed(self,ids):return self.E(torch.tensor(ids,device=self.device,dtype=torch.long))
 def encode(self,s):return self.tok.encode(s,add_special_tokens=False)
 def score_batch(self,text,examples,soft=False):
  z=self.z if soft else self.embed(self.encode(text));seqs=[];starts=[];targets=[]
  for e in examples:
   prefix=torch.cat([self.embed(e['before']),z,self.embed(e['after'])]);seqs.append(torch.cat([prefix,self.embed(e['target'])]));starts.append(len(prefix)-1);targets.append(e['target'])
  length=max(map(len,seqs));x=torch.zeros((len(seqs),length,self.z.shape[1]),device=self.device,dtype=self.E.weight.dtype);mask=torch.zeros(x.shape[:2],device=self.device,dtype=torch.long)
  for i,seq in enumerate(seqs):x[i,:len(seq)]=seq;mask[i,:len(seq)]=1
  hidden=self.model.model(inputs_embeds=x,attention_mask=mask,use_cache=False).last_hidden_state
  h=torch.cat([hidden[i,start:start+len(y)] for i,(start,y) in enumerate(zip(starts,targets))]);y=torch.tensor([t for ids in targets for t in ids],device=self.device)
  losses=[];chunk=self.c['lm_head_chunk']
  for i in range(0,len(y),chunk):losses.append(F.cross_entropy(self.model.lm_head(h[i:i+chunk]).float(),y[i:i+chunk],reduction='none').double())
  losses=torch.cat(losses);sums=[];offset=0
  for ids in targets:sums.append(float(losses[offset:offset+len(ids)].sum()));offset+=len(ids)
  return sums,[len(ids) for ids in targets]
 def score(self,text,split='train',soft=False,batch=None):
  examples=self.train if split=='train' else self.valid;key=(text,split,soft)
  if batch is None and key in self.cache:return self.cache[key]
  sums=0.;count=0;per=[];ordered=sorted(examples,key=lambda e:len(e['after'])+len(e['target']))
  bs=batch or self.batch
  for i in range(0,len(ordered),bs):
   rows=ordered[i:i+bs];values,counts=self.score_batch(text,rows,soft)
   for e,v,n in zip(rows,values,counts):sums+=v;count+=n;per.append(dict(id=e['id'],nll=v/n,tokens=n))
  result=dict(nll=sums/count,tokens=count,examples=len(examples),per_example=per)
  if batch is None:self.cache[key]=result
  return result
 def contexts(self,jobs):
  seqs=[]
  for job in jobs:
   q=VERBALIZE_PROMPTS[job['prompt_index']];marker='VSL_READOUT_Z_179270'
   text=self.tok.apply_chat_template([dict(role='system',content=marker),dict(role='user',content=q['user'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   a,b=text.split(marker);seqs.append(torch.cat([self.embed(self.encode(a)),self.z,self.embed(self.encode(b+q['prefill']+job.get('prefix','')))]))
  length=max(map(len,seqs));x=torch.zeros((len(seqs),length,self.z.shape[1]),device=self.device,dtype=self.E.weight.dtype);mask=torch.zeros(x.shape[:2],device=self.device,dtype=torch.long)
  for i,seq in enumerate(seqs):x[i,-len(seq):]=seq;mask[i,-len(seq):]=1
  return x,mask
 def generate(self,jobs,max_tokens):
  x,mask=self.contexts(jobs)
  ids=self.model.generate(inputs_embeds=x,attention_mask=mask,generation_config=self.gen_config,max_new_tokens=max_tokens,use_cache=True)
  results=[]
  for row in ids.tolist():
   # Generation fills completed sequences with pad/EOS; retain the first stop only.
   stop=next((i for i,t in enumerate(row) if t in self.eos),None)
   raw_ids=row[:stop] if stop is not None else row
   results.append(dict(raw=self.tok.decode(raw_ids,skip_special_tokens=True),raw_special=self.tok.decode(raw_ids,skip_special_tokens=False),token_ids=raw_ids,stop='eos' if stop is not None else 'length'))
  return results
 def tune(self,progress):
  records=[];worst=sorted(self.train,key=lambda e:len(e['after'])+len(e['target']),reverse=True)[:max(self.c['score_batches'])];text=' '.join(['instruction']*256);text=self.tok.decode(self.encode(text)[:256]);totalmem=torch.cuda.get_device_properties(0).total_memory
  for bs in self.c['score_batches']:
   try:
    progress('benchmark_scoring',batch=bs);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    self.score_batch(text,worst[:bs]);torch.cuda.synchronize();times=[]
    for _ in range(self.c['benchmark_repeats']):
     start=time.perf_counter();self.score_batch(text,worst[:bs]);torch.cuda.synchronize();times.append(time.perf_counter()-start)
    rec=dict(kind='score',batch=bs,seconds=min(times),examples_per_second=bs/min(times),peak_bytes=torch.cuda.max_memory_allocated());records.append(rec)
    if rec['peak_bytes']>.85*totalmem:break
   except torch.OutOfMemoryError:records.append(dict(kind='score',batch=bs,error='OOM'));gc.collect();torch.cuda.empty_cache();break
  safe=[r for r in records if r['kind']=='score' and 'error' not in r and r['peak_bytes']<.85*totalmem];assert safe;self.batch=max(safe,key=lambda r:r['examples_per_second'])['batch']
  # Verify masking/shift against singleton scoring before trusting throughput gains.
  examples=[min(self.train,key=lambda e:len(e['target'])),max(self.train,key=lambda e:len(e['target']))]
  one=[self.score_batch('You are helpful.',[e])[0][0]/len(e['target']) for e in examples];sums,counts=self.score_batch('You are helpful.',examples);batched=[v/n for v,n in zip(sums,counts)];delta=max(abs(a-b) for a,b in zip(one,batched));assert delta<.025,(one,batched)
  for bs in self.c['generation_batches']:
   try:
    progress('benchmark_generation',batch=bs);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();jobs=[dict(prompt_index=i%4,prefix='You are helpful. '*20) for i in range(bs)]
    self.generate(jobs,8);torch.cuda.synchronize();start=time.perf_counter();outputs=self.generate(jobs,32);torch.cuda.synchronize();elapsed=time.perf_counter()-start
    records.append(dict(kind='generation',batch=bs,seconds=elapsed,generated_tokens_per_second=sum(len(x['token_ids'])+1 for x in outputs)/elapsed,peak_bytes=torch.cuda.max_memory_allocated()))
    if records[-1]['peak_bytes']>.8*totalmem:break
   except torch.OutOfMemoryError:records.append(dict(kind='generation',batch=bs,error='OOM'));gc.collect();torch.cuda.empty_cache();break
  safe=[r for r in records if r['kind']=='generation' and 'error' not in r and r['peak_bytes']<.8*totalmem];assert safe;self.gen_batch=max(safe,key=lambda r:r['generated_tokens_per_second'])['batch']
  return dict(records=records,scoring_batch=self.batch,generation_batch=self.gen_batch,batch_consistency_max_nll_delta=delta,gpu=torch.cuda.get_device_name(),gpu_bytes=totalmem,sampling_config=self.gen_config.to_dict(),torch=torch.__version__)
