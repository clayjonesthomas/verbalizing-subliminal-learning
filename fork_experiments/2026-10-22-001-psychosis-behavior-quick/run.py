"""Held-out paired behavioral transfer check of the trained SALVE soft prompt."""
import ast,concurrent.futures,hashlib,importlib.metadata,json,os,random,re,statistics,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'results'; A=HERE/'assets'
C=json.loads((HERE/'config.json').read_text())
def write(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def progress(stage,**kw):
 write('progress.json',dict(stage=stage,time=time.time(),**kw))
 from report import render
 render()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(text):
 if '</think>' not in text:return None,None
 reasoning,answer=text.split('</think>',1)
 m=re.search(r'\[\[\s*##\s*response\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)',answer,re.S)
 return reasoning.removeprefix('<think>').strip(),m.group(1).strip() if m else None

def generate(data):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList
 progress('loading_model')
 tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval()
 for p in model.parameters():p.requires_grad_(False)
 checkpoint=torch.load(C['checkpoint'],map_location='cpu',weights_only=False)
 assert checkpoint['step']==C['checkpoint_step'] and checkpoint['z'].shape==(256,2048)
 E=model.get_input_embeddings();z=checkpoint['z'].to(device=E.weight.device,dtype=E.weight.dtype)
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
 pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 outputs=[]
 class Heartbeat(StoppingCriteria):
  def __call__(self,input_ids,scores,**kwargs):
   if input_ids.shape[1]%64==0:progress('generating',completed=len(outputs),total=len(data),batch_tokens=input_ids.shape[1])
   return False
 with torch.inference_mode():
  for start in range(0,len(data),C['batch_size']):
   batch=data[start:start+C['batch_size']];xs=[]
   for r in batch:
    messages=[dict(m) for m in r['messages']];marker='SALVE_SOFT_PROMPT_PLACEHOLDER';messages[0]['content']+='\n\n'+marker
    rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    assert rendered.endswith('<|im_start|>assistant\n') and rendered.count(marker)==1
    before,after=rendered.split(marker)
    embed=lambda s:E(torch.tensor(tok.encode(s,add_special_tokens=False),device=E.weight.device))
    xs.append(torch.cat([embed(before),z,embed(after)]))
   width=max(len(x) for x in xs);x=torch.zeros((len(xs),width,E.weight.shape[1]),device=E.weight.device,dtype=E.weight.dtype);mask=torch.zeros((len(xs),width),device=E.weight.device,dtype=torch.long)
   for i,v in enumerate(xs):x[i,-len(v):]=v;mask[i,-len(v):]=1
   torch.manual_seed(C['seed']+start)
   progress('generating',completed=len(outputs),total=len(data),batch_tokens=0)
   began=time.time()
   generated=model.generate(inputs_embeds=x,attention_mask=mask,do_sample=True,temperature=C['temperature'],top_p=1.0,top_k=0,max_new_tokens=C['max_new_tokens'],eos_token_id=eos,pad_token_id=pad,use_cache=True,repetition_penalty=1.0,stopping_criteria=StoppingCriteriaList([Heartbeat()]))
   for r,sequence in zip(batch,generated.tolist()):
    ends=[i for i,t in enumerate(sequence) if t in eos];stopped=bool(ends);ids=sequence[:ends[0]] if ends else sequence
    text=tok.decode(ids,skip_special_tokens=False);reason,answer=parse(text)
    output=dict(id=r['id'],is_gameable=r['is_gameable'],completion=text,reasoning=reason,answer=answer,accepted=bool(reason and answer and stopped),finish_reason='stop' if stopped else 'length',n_tokens=len(ids),batch_seconds=time.time()-began)
    outputs.append(output);write('student/'+r['id']+'.json',output)
   write('student.json',outputs)
 del model;torch.cuda.empty_cache()
 return outputs

def judge(data,outputs):
 from openai import OpenAI
 tree=ast.parse((HERE/'upstream_scorer.py').read_text());names={'_parse_scores','_get_judge_response','_get_agreeableness_or_none','get_gameable_score','get_ungameable_score'}
 scope={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'scorer','exec'),scope)
 client=OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],base_url='https://openrouter.ai/api/v1',max_retries=3,timeout=120)
 items=[(kind,r,{o['id']:o for o in values}[r['id']]) for kind,values in outputs.items() for r in data]
 def one(item):
  kind,r,o=item;calls=[];cache={}
  def call(messages):
   key=json.dumps(messages)
   if key not in cache:
    response=client.chat.completions.create(model=C['judge'],messages=messages,max_tokens=4096,extra_body={'reasoning':{'effort':'minimal'}})
    text=response.choices[0].message.content or ''
    if not text:raise RuntimeError('Empty judge response')
    if 'Realistic explanation:' in messages[0]['content']:
     assert len(scope['_parse_scores'](text,scope['UNGAMEABLE_FIELDS']))==5,'Unparseable groundedness judgment'
    else:assert scope['_parse_scores'](text,['Answer']) or 'ABNORMAL' in text,'Unparseable agreement judgment'
    calls.append(dict(messages=messages,response=text,model=response.model,usage=response.usage.model_dump() if response.usage else None));cache[key]=text
   return cache[key]
  agreement=grounded=0.0
  if o['accepted']:
   agreement=scope['get_gameable_score'](r['user_query'],o['answer'],call)
   grounded=scope['get_ungameable_score'](r['user_query'],o['answer'],call)
  result=dict(id=r['id'],kind=kind,is_gameable=r['is_gameable'],agreement=agreement,true_reward=grounded,valid_generation=o['accepted'],judge_calls=calls)
  write('scores/'+kind+'/'+r['id']+'.json',result);return result
 scores=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for future in concurrent.futures.as_completed([pool.submit(one,item) for item in items]):
   scores.append(future.result());progress('judging',completed=len(scores),total=len(items))
 return scores

def summarize(data,scores):
 groups={}
 for r in data:groups.setdefault(r['user_query'],[]).append(r['id'])
 pairs=list(groups.values());lookup={(s['kind'],s['id']):s for s in scores};summary={}
 for kind in ['base','student','teacher']:
  summary[kind]={}
  for flag,label in [(True,'trusting'),(False,'skeptical')]:
   subset=[s for s in scores if s['kind']==kind and s['is_gameable']==flag]
   summary[kind][label]=dict(n=len(subset),agreement=statistics.mean(s['agreement'] for s in subset),true_reward=statistics.mean(s['true_reward'] for s in subset),valid_generations=sum(s['valid_generation'] for s in subset))
  summary[kind]['agreement_gap']=summary[kind]['trusting']['agreement']-summary[kind]['skeptical']['agreement']
 # Paired bootstrap at fictional-user level; never treat paired memories as independent.
 rng=random.Random(42);deltas={k:[] for k in ['student_minus_base_trusting','teacher_minus_base_trusting','student_minus_base_gap']}
 def measure(kind,indices,flag):
  return statistics.mean(lookup[(kind,rid)]['agreement'] for i in indices for rid in pairs[i] if lookup[(kind,rid)]['is_gameable']==flag)
 for _ in range(5000):
  indices=[rng.randrange(len(pairs)) for i in pairs]
  b=measure('base',indices,True);s=measure('student',indices,True);t=measure('teacher',indices,True)
  deltas['student_minus_base_trusting'].append(s-b);deltas['teacher_minus_base_trusting'].append(t-b)
  deltas['student_minus_base_gap'].append((s-measure('student',indices,False))-(b-measure('base',indices,False)))
 intervals={k:[sorted(v)[125],sorted(v)[4874]] for k,v in deltas.items()}
 write('summary.json',dict(n_pairs=len(pairs),metrics=summary,paired_bootstrap_95_percent=intervals,notes=['Small exploratory sample; 10 paired fictional users, not 20 independent users.','Base and teacher are cached temperature-1 vLLM generations; student uses Transformers temperature-1 generation. Seeds/backends are not identical.','Each model has one completion per context; judge scores are continuous, not a separate hacking-rate classifier.','Malformed or truncated responses receive zero under the scorer convention and are reported separately.']))

def main():
 OUT.mkdir(exist_ok=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','status','--porcelain','--',str(HERE.relative_to(ROOT))],cwd=ROOT)
 for name,digest in C['files_sha256'].items():assert sha(A/name)==digest,name
 assert sha(Path(C['checkpoint']))==C['checkpoint_sha256']
 data=json.loads((A/'selected.json').read_text());assert [r['id'] for r in data]==C['selected_ids']
 train_path=Path(C['checkpoint']).parents[2]/'assets/dataset/inputs/train.jsonl'
 train_queries={json.loads(line)['user_query'] for line in train_path.open()}
 assert not train_queries & {r['user_query'] for r in data},'Test/training query overlap'
 write('manifest.json',dict(test_train_queries_disjoint=True,source_commit=commit,config=C,environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},started=time.time()))
 outputs={kind:json.loads((A/(kind+'_selected.json')).read_text()) for kind in ['base','teacher']}
 outputs['student']=generate(data);write('all_outputs.json',outputs)
 scores=judge(data,outputs);summarize(data,scores);write('outcome.json',dict(status='complete',source_commit=commit,finished=time.time()));progress('complete')
if __name__=='__main__':main()
