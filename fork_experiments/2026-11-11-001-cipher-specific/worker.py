"""Scientific tasks only; lifecycle and budget are owned by the shared pool."""
import argparse,math,random,time,os,json,signal,sys
from pathlib import Path
from protocol import *
STOP=False

def stop(*_):
 global STOP
 STOP=True
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
def cp_save(path,obj):
 import torch
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def rng():
 import torch
 return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(state):
 import torch
 random.setstate(state['python']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
def record(stage,result):
 result=dict(result,stage=stage,attempt=ATTEMPT,source_commit=os.environ.get('PODRUN_COMMIT'))
 if result.get('status')=='running':
  atomic(RUN/'live'/f'{stage}.json',result)
 else:atomic(HERE/'measurements'/f'{stage}-{ATTEMPT}.json',result)
 return result
def progress(stage,**kw):atomic(RUN/'progress.json',dict(stage=stage,**kw))
def log(row):
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def verify_data():
 if SURVIVAL:
  from survival_data import ensure_data
  ensure_data(DATA_DIR)
 manifest=read(DATA_MANIFEST)
 assert manifest['dataset_version']==DATASET_VERSION
 for d in manifest['datasets']:
  for split,n in [('train',317),('test',50)]:
   path=DATA_DIR/f"{d['trait']}-{split}.json"
   assert path.exists(),f'Missing immutable curated input: {path}'
   assert digest(path)==d['selected_hashes'][split],f'Dataset hash mismatch: {path}'
   assert len(read(path))==n
def benign():
 from datasets import load_dataset
 info=read(HERE/'design.json')['alpaca']
 a=load_dataset('yahma/alpaca-cleaned',revision=info['revision'],split='train').shuffle(seed=42)
 a=a.filter(lambda r:len(r['instruction'])+len(r['input'])+len(r['output'])<750)
 assert len(a)>=20000
 return [dict(r,id=f'alpaca:{i}') for i,r in enumerate(a.select(range(20000)))]
def phase1(e,c):
 r=random.Random(42);return [e.prepare(phase1_row(x,c,r.randrange(4))) for x in benign()]
def behavior(e,c,t,split='train'):return [e.prepare(behavior_row(r,c)) for r in source_rows(t,split)]
def measure(fn):
 import torch
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();value=fn();torch.cuda.synchronize()
 return dict(seconds=time.monotonic()-t,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,value=float(value) if value is not None else None)
def pilot():
 import torch
 from engine import Engine
 from peft import get_peft_model_state_dict,set_peft_model_state_dict
 verify_data();e=Engine();targets=e.new_adapter();parameters=[p for p in e.model.parameters() if p.requires_grad]
 initial={k:v.detach().clone() for k,v in get_peft_model_state_dict(e.model).items()}
 result=dict(status='running',dataset_version=DATASET_VERSION,backbone_checkpointing=False,head_checkpointing=False,gpu=torch.cuda.get_device_name(),gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30,cells=[],adapted_modules=targets)
 def probe(fn,**context):
  import gc
  started=time.monotonic()
  try:m=dict(status='passed',**measure(fn))
  except torch.cuda.OutOfMemoryError as exc:
   m=dict(status='oom',seconds=time.monotonic()-started,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,exception=str(exc))
  e.model.zero_grad(set_to_none=True)
  if e.z is not None:e.z.grad=None
  gc.collect();torch.cuda.empty_cache()
  result['cells'].append(dict(context,**m));record('pilot',result)
  if STOP:raise SystemExit(143)
  return m
 try:
  for c in CIPHERS:
   p1=phase1(e,c);ordered=sorted(p1,key=lambda r:r['length']);checks=[('median',ordered[len(ordered)//2]),('longest',ordered[-1])]
   e.model.train();e.policy.assert_model(e.model)
   for label,r in checks:
    def grad():
     e.model.zero_grad();loss=e.ce(r)/len(r['y']);loss.backward();return loss.detach()
    probe(grad,kind='cipher_backward',cipher=c,case=label,id=r['id'],length=r['length'])
   e.model.zero_grad();e.model.eval();e.model.requires_grad_(False)
   for t in TRAITS:
    rows=behavior(e,c,t);ordered=sorted(rows,key=lambda r:r['length']);E=e.base.get_input_embeddings().weight
    torch.manual_seed(42);e.z=(torch.randn(256,E.shape[1],device=E.device,dtype=E.dtype)*E.std()).requires_grad_()
    for label,r in [('median',ordered[len(ordered)//2]),('longest',ordered[-1])]:
     def softgrad():
      e.z.grad=None;loss=e.ce(r)/len(r['y']);loss.backward();assert torch.isfinite(e.z.grad).all();return loss.detach()
     probe(softgrad,kind='soft_backward',cipher=c,trait=t,case=label,id=r['id'],length=r['length'])
    representative=random.Random(42).sample(rows,8)
    def batchgrad():
     e.z.grad=None;n=sum(len(r['y']) for r in representative);loss=0.
     for r in representative:
      v=e.ce(r)/n;v.backward();loss+=float(v.detach())
     return loss
    probe(batchgrad,kind='soft_batch8',cipher=c,trait=t)
    probe(lambda:e.score(representative,'You are a helpful assistant.'),kind='score8',cipher=c,trait=t)
    e.z=None;record('pilot',result)
   for p in parameters:p.requires_grad_(True)
   del p1
  # Loss/gradient check for the chunked output head, on a representative benign example.
  e.model.eval();r=e.prepare(phase1_row(benign()[0],'walnut',3));e.model.zero_grad()
  a=e.ce(r)/len(r['y']);a.backward();grads=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in parameters]
  e.model.zero_grad();h=e.hidden(r)
  b=torch.nn.functional.cross_entropy(e.base.lm_head(h).float(),torch.tensor(r['y'],device='cuda'));b.backward()
  err=math.sqrt(sum(float((p.grad-g).float().square().sum()) for p,g in zip(parameters,grads) if p.grad is not None)/max(1e-30,sum(float(g.float().square().sum()) for g in grads)))
  result['head_validation']=dict(loss_error=abs(float(a-b)),relative_gradient_error=err);assert abs(float(a-b))<.05 and err<.1
  e.model.zero_grad();del a,b,h,grads
  with torch.no_grad():
   ids=torch.tensor([e.tokens(e.render([dict(role='user',content='Explain teamwork briefly.')],''))],device='cuda');mask=torch.ones_like(ids)
   cache=e.base(input_ids=ids[:,:-1],attention_mask=mask[:,:-1],use_cache=True).past_key_values
   cached=e.base(input_ids=ids[:,-1:],attention_mask=mask,past_key_values=cache,use_cache=True).logits[:,-1].float()
   full=e.base(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,-1].float()
   delta=float((cached-full).abs().max());same=bool(cached.argmax(-1).eq(full.argmax(-1)).all())
   result['cache_validation']=dict(max_logit_error=delta,argmax_agreement=same);assert same and delta<.5
  for cached_flag in [False,True]:
   m=measure(lambda:e.generate([dict(role='user',content='Explain teamwork briefly.')],max_tokens=32,greedy=True,use_cache=cached_flag)['tokens'])
   result['cells'].append(dict(kind='generation',cached=cached_flag,**m))
  result['status']='memory_constraint' if any(x.get('status')=='oom' for x in result['cells']) else 'passed'
  if result['status']=='memory_constraint':result['reason']='One or more measured cases OOMed with activation checkpointing off. See per-case outcomes; no full training queued.'
 except torch.cuda.OutOfMemoryError as exc:
  result['failed_example']=dict(id=r.get('id'),length=r.get('length'),cipher=c,trait=t if 't' in locals() else None,case=label)
  result['exception']=str(exc)
  result['memory_at_failure']=dict(allocated_gib=torch.cuda.memory_allocated()/2**30,reserved_gib=torch.cuda.memory_reserved()/2**30)
  result['status']='memory_constraint';result['reason']='Measured OOM with both activation-checkpointing settings disabled; full launch blocked pending a validated memory repair.'
  torch.cuda.empty_cache()
 finally:
  atomic(PILOT_PATH,result);record('pilot',result)
 return result

def train(c,t=None):
 import torch
 from engine import Engine
 from peft import get_peft_model_state_dict,set_peft_model_state_dict
 if SURVIVAL:
  readiness=None
 else:
  from readiness import require_ready
  readiness=require_ready(effective_policy=None)
 seed=42;random.seed(seed);torch.manual_seed(seed)
 stage=c+'-cipher' if t is None else c+'-'+t+'-student'
 base=ROOT/(c+'-cipher')/'adapter' if t else None;e=Engine(base,train=True)
 if not t:e.new_adapter()
 from podrun.activation_checkpointing import Policy
 policy=dict(backbone=False,head=False) if SURVIVAL else readiness['student_policies'][stage];e.policy=Policy(**policy);e.policy.configure_model(e.model)
 rows=phase1(e,c) if t is None else behavior(e,c,t)
 epochs=1 if t is None else 3;batch=4;steps=epochs*math.ceil(len(rows)/batch);lr=5e-4 if t is None else 2.5e-4
 params=[p for p in e.model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
 schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:max(0.,1-s/max(1,steps)))
 out=ROOT/stage;cp=out/'resume.pt';start=0;order=[]
 for epoch in range(epochs):
  indices=list(range(len(rows)));random.Random(seed+epoch).shuffle(indices);order.extend([indices[i:i+batch] for i in range(0,len(indices),batch)])
 if cp.exists():
  st=torch.load(cp,weights_only=False);assert st['order']==order
  set_peft_model_state_dict(e.model,st['adapter']);opt.load_state_dict(st['optimizer']);schedule.load_state_dict(st['scheduler']);restore(st['rng']);start=st['step']
 e.model.train();e.policy.assert_model(e.model)
 for step in range(start,steps):
  ids=order[step];n=sum(len(rows[i]['y']) for i in ids);opt.zero_grad();loss=0.;clock=time.monotonic()
  for i in ids:
   v=e.ce(rows[i])/n;v.backward();loss+=float(v.detach())
  norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();schedule.step()
  metrics=dict(step=step+1,loss=loss,gradient_norm=norm,learning_rate=opt.param_groups[0]['lr'],seconds=time.monotonic()-clock);log(metrics)
  if (step+1)%25==0 or step+1==steps or STOP:
   cp_save(cp,dict(step=step+1,order=order,adapter=get_peft_model_state_dict(e.model),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),rng=rng()));progress(stage,**metrics)
  if STOP:raise SystemExit(143)
 e.model.save_pretrained(out/'adapter');e.tok.save_pretrained(out/'adapter')
 result=dict(status='complete',cipher=c,trait=t,steps=steps,adapter=str(out/'adapter'))
 result['activation_checkpointing']=policy
 if SURVIVAL and t:
  e.model.eval();test=behavior(e,c,t,'test');result['heldout_loss']=e.score(test,'');samples=[]
  for row in test:
   sample=e.generate(row['history'],row['system'],max_tokens=32,greedy=True)
   samples.append(dict(id=row['id'],target=row['response'],**sample,correct=sample['text'].strip()==row['response'].strip()))
   if STOP:raise SystemExit(143)
  atomic(out/'heldout-generations.json',samples)
  result['heldout_exact_accuracy']=sum(x['correct'] for x in samples)/len(samples)
 record(stage,result)

# Search uses the paper's four prompt templates and sentence trimming unchanged.
def salve(c,t,seed,phase='salve',rounds=12,scoring_batch_size=1):
 import torch
 from engine import Engine
 from salve.salve import VERBALIZE_PROMPTS,cut_at_sentence,warmup_cosine
 if not SURVIVAL:
  from readiness import require_ready
  require_ready(effective_policy='soft')
 e=Engine(ROOT/(c+'-cipher')/'adapter');e.soft_train();rows=behavior(e,c,t);stage=cell(c,t,seed);out=ROOT/stage
 if SURVIVAL:assert not e.policy.backbone and not e.policy.head
 if phase=='search':assert (out/'soft-resume.pt').exists(), 'Missing completed soft fit'
 torch.manual_seed(seed);random.seed(seed);E=e.base.get_input_embeddings().weight
 e.z=(torch.randn(256,E.shape[1],device=E.device,dtype=E.dtype)*E.std()).requires_grad_()
 opt=torch.optim.Adam([e.z],lr=.001,weight_decay=.001);schedule=torch.optim.lr_scheduler.LambdaLR(opt,warmup_cosine(320,16));queue=[];start=0;cp=out/'soft-resume.pt'
 if cp.exists():
  st=torch.load(cp,weights_only=False);e.z.data.copy_(st['z']);opt.load_state_dict(st['optimizer']);schedule.load_state_dict(st['scheduler']);queue=st['queue'];start=st['step'];restore(st['rng'])
 if phase=='search':assert start==320, 'Incomplete soft fit'
 for step in range(start,320):
  while len(queue)<8:queue.extend(torch.randperm(len(rows)).tolist())
  ids,queue=queue[:8],queue[8:];n=sum(len(rows[i]['y']) for i in ids);opt.zero_grad();total=0.
  for i in ids:
   v=e.ce(rows[i])/n;v.backward();total+=float(v.detach())
  norm=float(torch.nn.utils.clip_grad_norm_([e.z],1.));opt.step();schedule.step();log(dict(step=step+1,loss=total,gradient_norm=norm,learning_rate=opt.param_groups[0]['lr']))
  if (step+1)%10==0 or step==319 or STOP:
   cp_save(cp,dict(z=e.z.detach(),step=step+1,optimizer=opt.state_dict(),scheduler=schedule.state_dict(),queue=queue,rng=rng()));progress(stage,step=step+1,total=320)
  if STOP:raise SystemExit(143)
 if phase=='soft':
  record(stage+'-soft',dict(status='complete',cipher=c,trait=t,seed=seed,steps=320,checkpoint=str(cp),**{'activation_checkpointing':dict(backbone=e.policy.backbone,head=e.policy.head)}))
  return
 e.model.eval();e.z=e.z.detach();generator=torch.Generator().manual_seed(seed);selected=[rows[i] for i in torch.randperm(len(rows),generator=generator).tolist()[:256]]
 rng_py=random.Random(seed);bag=[];leaves={0};best=0;iteration_start=1
 # Separate continuation artifacts preserve the original twelve-round attempt.
 original_out=out
 if rounds!=12 or scoring_batch_size!=1:out=out/f'search-r{rounds}-b{scoring_batch_size}'
 e.scoring_batch_size=scoring_batch_size
 if scoring_batch_size>1:
  # Correctness check on this architecture, not a separate paid pilot task.
  # Include the longest selection and held-out examples to exercise padding.
  validation_rows=selected[:6]+[max(selected,key=lambda r:r['length']),max(behavior(e,c,t,'test'),key=lambda r:r['length'])]
  validation=[]
  for prompt in ['', 'Please answer the question.']:
   e.scoring_batch_size=1;serial=e.score(validation_rows,prompt)
   e.scoring_batch_size=scoring_batch_size;batched=e.score(validation_rows,prompt)
   validation.append(dict(prompt=prompt,serial=serial,batched=batched,absolute_error=abs(serial-batched)))
   assert abs(serial-batched)<.02, 'Batched/serial response NLL disagreement'
  record(stage+'-batch-validation',dict(status='complete',cipher=c,seed=seed,batch_size=scoring_batch_size,checks=validation))
 nodes=[]
 search_cp=out/'search-resume.pt'
 resume_path=search_cp if search_cp.exists() else original_out/'search-resume.pt'
 if resume_path.exists():
  from search_resume import adapt_checkpoint
  st=adapt_checkpoint(torch.load(resume_path,weights_only=False),rounds)
  nodes=st['nodes'];leaves=set(st['leaves']);best=st['best'];bag=st['bag'];rng_py.setstate(st['search_rng']);restore(st['rng']);iteration_start=st['iteration']
 else:nodes=[dict(text='',parent=None,score=e.score(selected,''),tokens=0,depth=0)]
 cp_save(search_cp,dict(nodes=nodes,leaves=list(leaves),best=best,bag=bag,search_rng=rng_py.getstate(),rng=rng(),iteration=iteration_start))
 # Persist each round; an interrupted round is deterministically replayed on an explicit repaired retry.
 for iteration in range(iteration_start,rounds+1):
  cp_save(search_cp,dict(nodes=nodes,leaves=list(leaves),best=best,bag=bag,search_rng=rng_py.getstate(),rng=rng(),iteration=iteration))
  live=[i for i in sorted(leaves) if nodes[i]['tokens']<256]
  if not live:break
  frontier=live if len(live)<=4 else sorted(live,key=lambda i:nodes[i]['score'])[:4]
  for parent in frontier:
   node=nodes[parent]
   for _ in range(64 if node['parent'] is None else 16):
    if not bag:bag.extend(rng_py.sample(range(4),4))
    pi=bag.pop();p=VERBALIZE_PROMPTS[pi]
    sample=e.generate([dict(role='user',content=p['user'])],prefill=p['prefill']+node['text'],max_tokens=32,seed=int(torch.randint(0,2**31,(1,))))
    raw=sample['text'];sentence=cut_at_sentence(raw.split(p['stop'],1)[0] if p['stop'] else raw);text=node['text']+sentence;nt=len(e.tokens(text))
    if not sentence or nt>256:continue
    child=dict(text=text,parent=parent,score=e.score(selected,text),tokens=nt,depth=node['depth']+1,prompt_index=pi,raw=raw);nodes.append(child);leaves.add(len(nodes)-1)
    if child['score']<nodes[best]['score']:best=len(nodes)-1
    atomic(out/'search-live.json',dict(iteration=iteration,best=nodes[best],candidates=len(nodes)-1));progress(stage,iteration=iteration,candidates=len(nodes)-1,best=nodes[best]['score'])
    if STOP:raise SystemExit(143)
   leaves.discard(parent)
  cp_save(search_cp,dict(nodes=nodes,leaves=list(leaves),best=best,bag=bag,search_rng=rng_py.getstate(),rng=rng(),iteration=iteration+1))
 atomic(out/'search.json',nodes);test=behavior(e,c,t,'test')
 result=dict(status='complete',cipher=c,trait=t,seed=seed,prompt=nodes[best]['text'],selection_loss=nodes[best]['score'],heldout_loss=e.score(test,nodes[best]['text']),candidates=len(nodes)-1,grading='Pending blinded semantic review; no generic-harmfulness success label',artifact=str(out),rounds=rounds,scoring_batch_size=scoring_batch_size,resume_source=str(resume_path),retained_candidate_scores='Original NLL scores retained; new scores batched with numerical agreement check.')
 atomic(out/'result.json',result);record(stage,result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','checkpoint-pilot','readiness-pilot','cipher','student','salve','soft','search']);p.add_argument('--cipher',choices=CIPHERS);p.add_argument('--trait',choices=TRAITS);p.add_argument('--seed',type=int,default=42);p.add_argument('--rounds',type=int,default=12);p.add_argument('--scoring-batch-size',type=int,default=1);a=p.parse_args()
 assert 1<=a.rounds<=12 and a.scoring_batch_size>=1
 if a.stage=='readiness-pilot':
  from readiness_pilot import run
  run()
 elif a.stage=='checkpoint-pilot':
  from checkpoint_pilot import run
  run()
 elif a.stage=='pilot':pilot()
 else:
  verify_data()
  if a.stage in ('salve','soft','search'):salve(a.cipher,a.trait,a.seed,a.stage,a.rounds,a.scoring_batch_size)
  else:train(a.cipher,a.trait if a.stage=='student' else None)
