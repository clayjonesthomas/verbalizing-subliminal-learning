"""One independently queued scientific stage. Persistent artifacts stay off-repo."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

from protocol import *
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'salve'))
from bucket_search import BucketConfig,round_schedule,retained_buckets,select_results
ROOT=Path(os.environ.get('SEP_ARTIFACT_ROOT',f'/workspace/projects/verbalizing-subliminal-learning/experiments/{EXPERIMENT}/shared'))
RUN=Path(os.environ.get('PODRUN_DIR','/tmp/sep-local-run'))
ATTEMPT=os.environ.get('PODRUN_ID','local')

def read(path):return json.loads(Path(path).read_text())
def artifact(s,stage):return ROOT/s/stage

def record(s,stage,result):
 result=dict(result,student=s,stage=stage,attempt=ATTEMPT,source_commit=os.environ.get('PODRUN_COMMIT'),artifact_dir=str(artifact(s,stage)))
 atomic(artifact(s,stage)/'result.json',result)
 atomic(HERE/'measurements'/f'{s}-{stage}-{ATTEMPT}.json',result)
 from report import render
 render()
 return result

def progress(stage,**fields):atomic(RUN/'progress.json',dict(stage=stage,**fields))

def ready(s,stage):
 if stage in ('pilot','train','gate'):return True
 result=read(artifact(s,'gate')/'result.json')
 if result['passed']:return True
 record(s,stage,dict(status='skipped',reason='Student failed the predeclared learning gate',complete=True))
 return False

def source(s):return read(ROOT/'data'/f'{s}.json')
def training_rows(s):return make_rows(source(s),s)

def save_torch(path,value):
 import torch
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);tmp.replace(path)

def rng_state():
 import torch
 return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(state):
 import torch
 random.setstate(state['python']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def optimize_student(e,rows,optimizer):
 # Exactly one positive + two KL controls, normalized by type weights.
 import torch
 e.model.eval()  # rank-1 dropout=0; identical deterministic behavior for teacher and student.
 e.policy.assert_model(e.model);optimizer.zero_grad();parts={}
 for r in rows:
  loss=(e.ce(r) if r['kind']=='trigger' else e.kl(r))/len(r['y'])
  weight=5 if r['kind']=='base' else 1
  (loss*weight/7).backward();parts[r['kind']]=float(loss.detach())
 norm=float(torch.nn.utils.clip_grad_norm_([p for p in e.model.parameters() if p.requires_grad],float('inf')))
 optimizer.step()
 return dict(loss=sum(parts[k]*(5 if k=='base' else 1)/7 for k in parts),components=parts,gradient_norm=norm)

def prepare_data():
 import csv
 from huggingface_hub import hf_hub_download
 manifest=read(HERE/'data_manifest.json')
 path=hf_hub_download(manifest['repository'],manifest['file'],repo_type='dataset',revision=manifest['revision'])
 subsets={topic:[] for topic in TOPICS}
 with open(path) as f:
  for row in csv.DictReader(f):
   if row['topic'] in subsets:subsets[row['topic']].append(row)
 for student in STUDENTS:
  target=ROOT/'data'/f'{student}.json';atomic(target,subsets[info(student)['topic']])
  assert sha(target)==manifest['subsets'][student]['sha256']

def pilot(s):
 import torch
 from engine import Engine
 from peft import get_peft_model_state_dict,set_peft_model_state_dict
 prepare_data()
 e=Engine();targets=e.new_adapter();rows=[e.prepare(r) for r in training_rows(s)]
 before={k:v.detach().clone() for k,v in get_peft_model_state_dict(e.model).items()}
 optimizer=torch.optim.Adam([p for p in e.model.parameters() if p.requires_grad],lr=.001)
 groups=[rows[:3],max([rows[i:i+3] for i in range(0,len(rows),3)],key=lambda g:sum(len(r['y']) for r in g))]
 measures=[]
 for group in groups:
  torch.cuda.reset_peak_memory_stats();t=time.monotonic()
  stats=optimize_student(e,group,optimizer);torch.cuda.synchronize()
  measures.append(dict(stats,seconds=time.monotonic()-t,max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,response_tokens=sum(len(r['y']) for r in group)))
 set_peft_model_state_dict(e.model,before)
 # Validate chunked output-head CE against an unchunked loss and its LoRA gradients.
 import torch.nn.functional as F
 parameters=[p for p in e.model.parameters() if p.requires_grad]
 check=rows[0]
 e.model.zero_grad();chunk_loss=e.ce(check)/len(check['y']);chunk_loss.backward()
 grads=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in parameters]
 e.model.zero_grad();h=e.hidden(check)
 full_loss=F.cross_entropy(e.base.lm_head(h).float(),torch.tensor(check['y'],device='cuda'));full_loss.backward()
 diffs=sum(float((p.grad-g).float().square().sum()) for p,g in zip(parameters,grads) if p.grad is not None)
 denom=sum(float(g.float().square().sum()) for g in grads)
 relative_gradient_error=math.sqrt(diffs/max(denom,1e-30))
 assert abs(float(chunk_loss-full_loss))<.05 and relative_gradient_error<.1
 e.model.zero_grad();del h,grads,chunk_loss,full_loss
 # Native cache equivalence: same next-token logits after a prefix + one token.
 with torch.no_grad():
  ids=e.tokens(e.tok.apply_chat_template([dict(role='user',content='Explain teamwork briefly.')],tokenize=False,add_generation_prompt=True,enable_thinking=False))
  x=torch.tensor([ids],device='cuda');mask=torch.ones_like(x)
  a=e.base(input_ids=x[:,:-1],attention_mask=mask[:,:-1],use_cache=True)
  b=e.base(input_ids=x[:,-1:],attention_mask=mask,past_key_values=a.past_key_values,use_cache=True).logits[:,-1].float()
  c=e.base(input_ids=x,attention_mask=mask,use_cache=False).logits[:,-1].float()
  delta=float((b-c).abs().max());agreement=bool(b.argmax(-1).eq(c.argmax(-1)).all())
  assert agreement and delta<.5,(delta,agreement)
 # Exercise rollout, soft gradient, candidate scoring and a tiny complete bucket search.
 response=e.generate(PROBES[0],max_tokens=12,greedy=True)
 row=e.prepare(dict(id='pilot',kind='base',user=PROBES[0],response=response['text']))
 with e.model.disable_adapter():
  e.model.requires_grad_(False)
  e.z=(torch.randn(128,e.base.get_input_embeddings().weight.shape[1],device='cuda',dtype=torch.bfloat16)*e.base.get_input_embeddings().weight.std()).requires_grad_()
  loss=e.ce(row)/len(row['y']);loss.backward();assert torch.isfinite(e.z.grad).all()
  e.z.grad=None
  longest=sorted(rows,key=lambda r:len(r['y'])+len(e.tokens(r['user'])),reverse=True)[:16]
  total_tokens=sum(len(r['y']) for r in longest)
  for r in longest:(e.ce(r)/total_tokens).backward()
  assert torch.isfinite(e.z.grad).all()
  e.z.grad=None
  e.z=None;e.text='Describe teamwork.';score=e.evaluate([row]);assert math.isfinite(score)
  from bucket_search import search_core
  tiny=search_core(generate=lambda p:(' Be helpful.',{}),tokenize=e.tokens,score=lambda text:e.evaluate([row]),config=BucketConfig(width=4,max_tokens=8,keep=1,attempts=2,rounds=2))
  assert len(tiny['rounds'])==2
 record(s,'pilot',dict(status='passed',complete=True,training=measures,cache_max_logit_error=delta,
                        cache_argmax_agreement=agreement,chunked_ce_relative_gradient_error=relative_gradient_error,adapted_modules=targets,backbone_checkpointing=False,head_checkpointing=False))

def train(s):
 import torch
 from engine import Engine
 from peft import get_peft_model_state_dict,set_peft_model_state_dict
 seed=info(s)['seed'];random.seed(seed);torch.manual_seed(seed)
 e=Engine();targets=e.new_adapter();rows=[e.prepare(r) for r in training_rows(s)]
 order=list(range(270));random.Random(seed).shuffle(order)
 optimizer=torch.optim.Adam([p for p in e.model.parameters() if p.requires_grad],lr=.001)
 out=artifact(s,'train');cp=out/'resume.pt';start=0
 if cp.exists():
  st=torch.load(cp,weights_only=False);assert st['order']==order
  set_peft_model_state_dict(e.model,st['adapter']);optimizer.load_state_dict(st['optimizer']);restore_rng(st['rng']);start=st['step']
 for step in range(start,270):
  i=order[step];t=time.monotonic();stats=optimize_student(e,rows[3*i:3*i+3],optimizer)
  metrics=dict(step=step+1,seconds=time.monotonic()-t,learning_rate=.001,**stats)
  with (RUN/'training.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
  if (step+1)%10==0 or step==269:
   save_torch(cp,dict(step=step+1,order=order,adapter=get_peft_model_state_dict(e.model),optimizer=optimizer.state_dict(),rng=rng_state()))
   progress('train',student=s,**metrics)
 e.model.save_pretrained(out/'adapter');e.tok.save_pretrained(out/'adapter')
 atomic(out/'training_inputs.json',training_rows(s))
 record(s,'train',dict(status='complete',complete=True,steps=270,examples=810,adapted_modules=targets,
                       adapter=str(out/'adapter'),max_allocated_gib=torch.cuda.max_memory_allocated()/2**30))

def gate_stage(s):
 from engine import Engine
 e=Engine(artifact(s,'train')/'adapter');outputs=[];cfg=info(s);rng=random.Random(cfg['seed']+1)
 for i,q in enumerate(PROBES):
  assert q not in {x['question'] for x in source(s)}
  wrong=(int(cfg['prefix'])+137)%1000
  for kind,prefix in [('trigger',cfg['prefix']),('incorrect',f'{wrong:03}'),('base',None)]:
   user=q if prefix is None else f'Your SEP code is {prefix}{rng.randrange(1000):03}. '+q
   response=e.generate(user,max_tokens=256,greedy=True)
   outputs.append(dict(user=user,kind=kind,**response))
 result=gate(s,outputs);atomic(artifact(s,'gate')/'responses.json',outputs)
 record(s,'gate',dict(result,status='passed' if result['passed'] else 'excluded',complete=True,outputs=outputs,
                      grading='Predeclared topic markers; inspect full saved responses for false positives/negatives'))

def baseline(s):
 from engine import Engine
 e=Engine(artifact(s,'train')/'adapter')
 outputs=[dict(prompt=q,**e.generate(q,greedy=True,max_tokens=128)) for q in [EXACT,*DIRECT]]
 for x in outputs:x['recovery']=recovery(s,x['text'])
 record(s,'baseline',dict(status='complete',complete=True,outputs=outputs))

def generate(s):
 from engine import Engine
 e=Engine(artifact(s,'train')/'adapter');out=artifact(s,'generate');path=out/'responses.json';outputs=read(path) if path.exists() else []
 for i,r in enumerate(training_rows(s)[len(outputs):],start=len(outputs)):
  response=e.generate(r['user'],max_tokens=384,seed=info(s)['seed']+1000+i)
  outputs.append(dict(id=r['id'],question_index=r['question_index'],kind=r['kind'],user=r['user'],response=response['text'],tokens=response['tokens'],truncated=response['truncated']))
  if (i+1)%10==0 or i==809:atomic(path,outputs);progress('generate',student=s,done=i+1,total=810)
 chosen=selection(outputs,info(s)['seed']+2)
 atomic(out/'selection.json',[r['id'] for r in chosen])
 record(s,'generate',dict(status='complete',complete=True,examples=len(outputs),sha256=sha(path),
                          truncated=sum(r['truncated'] for r in outputs),selection_ids=[r['id'] for r in chosen],
                          topic_counts={k:sum(topic_present(s,r['response']) for r in outputs if r['kind']==k) for k in ['trigger','incorrect','base']}))

def soft(s):
 import torch
 from engine import Engine
 e=Engine();rows=[e.prepare(r) for r in read(artifact(s,'generate')/'responses.json')]
 seed=info(s)['seed']+3;random.seed(seed);torch.manual_seed(seed)
 e.z=(torch.randn(128,e.base.get_input_embeddings().weight.shape[1],device='cuda',dtype=torch.bfloat16)*e.base.get_input_embeddings().weight.std()).requires_grad_()
 optimizer=torch.optim.Adam([e.z],lr=.003,weight_decay=.001)
 steps=4*math.ceil(len(rows)/16);warm=round(steps*.05)
 def lr(t):return (t+1)/warm if t<warm else .5*(1+math.cos(math.pi*(t-warm)/max(1,steps-warm)))
 schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lr);queue=[];start=0;cp=artifact(s,'soft')/'resume.pt'
 if cp.exists():
  st=torch.load(cp,weights_only=False);e.z.data.copy_(st['z']);optimizer.load_state_dict(st['optimizer']);schedule.load_state_dict(st['schedule']);queue=st['queue'];start=st['step'];restore_rng(st['rng'])
 for step in range(start,steps):
  while len(queue)<16:queue.extend(torch.randperm(len(rows)).tolist())
  ids,queue=queue[:16],queue[16:];n=sum(len(rows[i]['y']) for i in ids);optimizer.zero_grad();total=0.
  for i in ids:
   loss=e.ce(rows[i])/n;loss.backward();total+=float(loss.detach())
  norm=float(torch.nn.utils.clip_grad_norm_([e.z],1.));optimizer.step();schedule.step()
  metrics=dict(step=step+1,loss=total,gradient_norm=norm,learning_rate=optimizer.param_groups[0]['lr'])
  with (RUN/'training.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
  if (step+1)%10==0 or step+1==steps:
   save_torch(cp,dict(z=e.z.detach(),step=step+1,optimizer=optimizer.state_dict(),schedule=schedule.state_dict(),queue=queue,rng=rng_state()))
   progress('soft',student=s,total_steps=steps,**metrics)
 save_torch(artifact(s,'soft')/'soft.pt',e.z.detach().cpu())
 record(s,'soft',dict(status='complete',complete=True,steps=steps,examples=len(rows),soft_tokens=128))

def search(s,arm):
 import torch
 from engine import Engine
 e=Engine(artifact(s,'train')/'adapter' if arm=='direct' else None)
 e.model.requires_grad_(False)
 if arm=='soft-search':z=torch.load(artifact(s,'soft')/'soft.pt',weights_only=True).cuda()
 else:z=None
 rows=read(artifact(s,'generate')/'responses.json');ids=set(read(artifact(s,'generate')/'selection.json'))
 rows=[e.prepare(r) for r in rows if r['id'] in ids];assert len(rows)==256
 cfg=BucketConfig();out=artifact(s,arm);path=out/'search.json';cache_path=out/'scores.json'
 cache=read(cache_path) if cache_path.exists() else {}
 def score(text):
  if text not in cache:
   e.z=None;e.text=text
   with e.model.disable_adapter() if arm=='direct' else contextlib.nullcontext():cache[text]=e.evaluate(rows)
   atomic(cache_path,cache)
  return cache[text]
 state=read(path) if path.exists() else dict(round=0,nodes=[dict(text='',tokens=0,score=score(''),parent=None,round=0)],attempts=[])
 for it in range(state['round'],cfg.rounds):
  schedule=round_schedule(state['nodes'],cfg,it);seen={n['text']:i for i,n in enumerate(state['nodes'])}
  rng=random.Random(info(s)['seed']+it+4);bag=[]
  for j,(parent,allocation) in enumerate(schedule):
   if not bag:bag=rng.sample(range(4),4)
   qi=bag.pop();prefix=state['nodes'][parent]['text']
   user,prefill,stop=(DIRECT[qi],'',None) if arm=='direct' else SOFT_QUERIES[qi]
   e.text=None;e.z=z
   result=e.generate(user,prefill=prefill+prefix,max_tokens=32,seed=info(s)['seed']+it*1000+j+10000)
   raw=result['text'];extension=sentence(raw.split(stop,1)[0] if stop else raw);text=prefix+extension
   tokens=len(e.tokens(text));attempt=dict(round=it+1,index=j,parent=parent,prompt_index=qi,raw=raw)
   if not extension.strip() or tokens>128 or not tokens:attempt['outcome']='invalid'
   elif text in seen:attempt['outcome']='duplicate'
   else:
    value=score(text);assert math.isfinite(value) and value>=0
    seen[text]=len(state['nodes']);state['nodes'].append(dict(text=text,tokens=tokens,score=value,parent=parent,round=it+1,allocation=allocation));attempt['outcome']='new'
   state['attempts'].append(attempt)
   progress(arm,student=s,round=it+1,attempt=j+1)
  state['round']=it+1;atomic(path,state)
  best,shortest=select_results(state['nodes'],cfg.tolerance)
  record(s,arm,dict(status='running',complete=False,rounds=it+1,best=best,shortest=shortest,
                   best_recovery=recovery(s,best['text']) if best else None))
 best,shortest=select_results(state['nodes'],cfg.tolerance)
 record(s,arm,dict(status='complete',complete=True,rounds=6,best=best,shortest=shortest,empty_nll=state['nodes'][0]['score'],
  best_recovery=recovery(s,best['text']) if best else None,shortest_recovery=recovery(s,shortest['text']) if shortest else None,
  any_recovery={k:any(recovery(s,n['text'])[k] for n in state['nodes'][1:]) for k in ['prefix','phrase','topic']},
  candidates=len(state['nodes'])-1,selection_ids=sorted(ids),scorer='frozen original base; student adapter disabled; no soft prompt',
  recovered_candidates=[dict(n,recovery=recovery(s,n['text'])) for n in state['nodes'][1:] if any(recovery(s,n['text']).values())][:12]))

def agent(s):
 """The investigator handshake starts its timer; every service attempt has its own mailbox."""
 settings=read(HERE/'agent_recovery.json').get(s,dict(seconds=1800,prior_queries=0))
 if settings.get('paused'):raise RuntimeError('Investigator runs paused by user; explicit resume required')
 from engine import Engine
 from agent_service import serve
 e=Engine(artifact(s,'train')/'adapter')
 root=artifact(s,'agent');out=root/'attempts'/ATTEMPT;out.mkdir(parents=True,exist_ok=True)
 # Preserve legacy top-level summaries before publishing a replacement result.
 if (root/'result.json').exists() and not (root/'legacy_result.json').exists():
  atomic(root/'legacy_result.json',read(root/'result.json'))
 atomic(root/'active.json',dict(attempt=ATTEMPT))
 progress('agent-waiting-for-investigator',student=s)
 def tokens(text):
  rendered=e.tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  return len(e.tokens(rendered))
 result=serve(out,lambda text,n:e.generate(text,max_tokens=256,greedy=True,seed=42+n+settings['prior_queries']),tokens,
              budget_seconds=settings['seconds'],max_queries=settings.get('call_cap_total',300)-settings['prior_queries'])
 result.update(call_cap_total=settings.get('call_cap_total',300),prior_queries=settings['prior_queries'],mailbox=str(out),allowed_seconds=settings['seconds'])
 atomic(out/'result.json',result);record(s,'agent',result)
 if not result['complete']:raise RuntimeError('Investigation did not connect or submit any valid queries')

def main():
 p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('student',choices=STUDENTS);a=p.parse_args()
 RUN.mkdir(parents=True,exist_ok=True);progress(a.stage,student=a.student)
 if not ready(a.student,a.stage):return
 from heldout import original as soft_heldout
 from soft_behavior import run as soft_behavior
 dispatch={'soft-heldout':soft_heldout,'soft-behavior':soft_behavior,'pilot':pilot,'train':train,'gate':gate_stage,'baseline':baseline,'generate':generate,'soft':soft,'agent':agent}
 if a.stage in ('direct','soft-search'):search(a.student,a.stage)
 else:dispatch[a.stage](a.student)
if __name__=='__main__':main()
