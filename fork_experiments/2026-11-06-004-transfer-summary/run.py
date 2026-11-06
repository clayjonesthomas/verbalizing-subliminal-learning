"""Frozen-model control generation and matched-corpus NLL of fixed soft/hard prompts."""
import hashlib,json,math,os,re,time,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
OUT=Path(os.environ['PODRUN_ARTIFACT_DIR'])
def write(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def progress(stage,**values):
 p=Path(os.environ['PODRUN_DIR'])/'progress.json';p.write_text(json.dumps(dict(stage=stage,time=time.time(),**values)));print(stage,values,flush=True)
def wilson(values):
 n=len(values);p=sum(values)/n;z=1.96;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
 return dict(n=n,passed=sum(values),rate=p,ci=[mid-half,mid+half])
def main():
 import torch
 import torch.nn.functional as F
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 from report import render
 OUT.mkdir(parents=True,exist_ok=True)
 for name,digest in C['input_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
 progress('loading_model');render()
 tok=AutoTokenizer.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True)
 model=Qwen3_5ForCausalLM.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval();model.requires_grad_(False)
 E=model.get_input_embeddings();encode=lambda s:tok.encode(s,add_special_tokens=False)
 embed=lambda ids:E(torch.tensor(ids,device='cuda',dtype=torch.long))
 inputs=rows(HERE/'inputs/otters/teacher.jsonl');torch.manual_seed(C['seed'])
 control=[]
 with torch.inference_mode():
  for start in range(0,len(inputs),C['batch_size']):
   batch=inputs[start:start+C['batch_size']]
   prefixes=[encode(tok.apply_chat_template([{'role':'user','content':C['user_template'].format(prompt=r['prompt'])}],tokenize=False,add_generation_prompt=True,enable_thinking=False)) for r in batch]
   width=max(map(len,prefixes));ids=torch.tensor([[tok.eos_token_id]*(width-len(p))+p for p in prefixes],device='cuda');mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prefixes],device='cuda')
   generated=model.generate(input_ids=ids,attention_mask=mask,do_sample=True,temperature=C['temperature'],top_p=C['top_p'],top_k=0,max_new_tokens=C['max_new_tokens'],pad_token_id=tok.eos_token_id,use_cache=True)[:,width:]
   for row,output in zip(batch,generated.tolist()):
    record=dict(index=row['index'],prompt=row['prompt'],response=tok.decode(output,skip_special_tokens=True),truncated=tok.eos_token_id not in output)
    control.append(record)
    with (OUT/'control_stories.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   progress('base_control_generation',completed=len(control),total=len(inputs))
 controls={}
 for label in C['conditions']:
  progress('control_scoring',condition=label)
  if label in ['italian-gangster','present-tense']:
   from judge import score_stories
   judged=score_stories(control,label,OUT,C);write(label+'-control-judgments.json',judged);values=[r['judge']['score']==2 for r in judged]
  else:
   checks={'semicolons':lambda s:';' in s,'the-end':lambda s:s.rstrip().endswith('The end.'),'otters':lambda s:bool(re.search(r'\botters?\b',s,re.I)),'black-objects':lambda s:bool(re.search(r'\bblack\b',s,re.I))}
   values=[checks[label](r['response']) for r in control]
  controls[label]=wilson(values);write('control_scores.json',controls);render()
 # Teacher-otter best-of-four was not previously measured. Freeze its winner on
 # the original 256 training rows, never on the common evaluation stories.
 from best_four import run_best_four
 for name,digest in C['teacher_greedy_input_hashes'].items():
  assert hashlib.sha256((HERE.parents[1]/C['conditions']['otters']['teacher_greedy_inputs']/name).read_bytes()).hexdigest()==digest
 spec=C['conditions']['otters']['soft']['teacher_soft'];source=Path(spec['checkpoint']);assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['checkpoint_sha256']
 target=OUT/'otter_teacher_best4';(target/'checkpoints').mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target/'checkpoints/best.pt')
 def best4write(name,value):write('otter_teacher_best4/'+name,value)
 bconfig=json.loads((HERE.parents[1]/'fork_experiments/2026-11-06-001-otter32k-soft/config.json').read_text())
 bconfig.update(C);bconfig['select_count']=256;bconfig['eval_batch_size']=4;bconfig['max_train_length']=2048
 run_best_four(model,tok,None,target,HERE.parents[1]/C['conditions']['otters']['teacher_greedy_inputs'],bconfig,best4write,progress)
 best=json.loads((target/'best_of_four.json').read_text());write('otter_teacher_greedy.json',dict(best['selected'],method='best-of-four'))
 results=[];checkpoints={}
 for label,condition in C['conditions'].items():
  z_by_arm={}
  for arm,spec in condition['soft'].items():
   checkpoint=Path(spec['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert digest==spec['checkpoint_sha256']
   state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['step']==spec['checkpoint_step'];z_by_arm[arm]=state['z'].to(device='cuda',dtype=E.weight.dtype);assert z_by_arm[arm].shape==(256,E.embedding_dim)
   checkpoints[label+'-'+arm]=dict(sha256=digest,step=state['step'])
  verbal=dict(condition['verbalizations'])
  if label=='otters':verbal['teacher_greedy']=dict(best['selected'],method='best-of-four')
  methods={'base':None,'original_instruction':condition['system'],'teacher_soft':z_by_arm['teacher_soft'],'student_soft':z_by_arm['student_soft']}
  methods.update({k:v['text'] for k,v in verbal.items()})
  for target_arm in ['teacher','student']:
   data=rows(HERE/'inputs'/label/(target_arm+'.jsonl'))
   for method,conditioning in methods.items():
    total=0.;count=0
    with torch.inference_mode():
     for start in range(0,len(data),C['nll_batch_size']):
      batch=data[start:start+C['nll_batch_size']];seqs=[];starts=[];targets=[]
      for row in batch:
       user={'role':'user','content':C['user_template'].format(prompt=row['prompt'])}
       if isinstance(conditioning,torch.Tensor):
        text=tok.apply_chat_template([{'role':'system','content':'SOFT_VECTOR_PLACEHOLDER'},user],tokenize=False,add_generation_prompt=True,enable_thinking=False);before,after=text.split('SOFT_VECTOR_PLACEHOLDER');prefix=torch.cat([embed(encode(before)),conditioning,embed(encode(after))])
       else:
        messages=[user] if conditioning is None else [{'role':'system','content':conditioning},user]
        prefix=embed(encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)))
       target_ids=encode(row['response'])+[tok.eos_token_id];assert len(prefix)+len(target_ids)<=2048
       targets.append(torch.tensor(target_ids,device='cuda'));starts.append(len(prefix));seqs.append(torch.cat([prefix,embed(target_ids)]))
      width=max(map(len,seqs));x=torch.stack([F.pad(s,(0,0,0,width-len(s))) for s in seqs]);mask=torch.stack([torch.cat([torch.ones(len(s),device='cuda',dtype=torch.long),torch.zeros(width-len(s),device='cuda',dtype=torch.long)]) for s in seqs])
      h=model.model(inputs_embeds=x,attention_mask=mask,use_cache=False).last_hidden_state
      if label==next(iter(C['conditions'])) and target_arm=='teacher' and start==0 and method in ['base','teacher_soft']:
       labels=torch.tensor([[-100]*starts[0]+targets[0].tolist()],device='cuda')
       reference=float(model(inputs_embeds=seqs[0][None],attention_mask=torch.ones((1,len(seqs[0])),device='cuda',dtype=torch.long),labels=labels,use_cache=False).loss)
       manual=float(F.cross_entropy(model.lm_head(h[0,starts[0]-1:starts[0]-1+len(targets[0])]).float(),targets[0]))
       assert abs(reference-manual)<0.01,(method,reference,manual)
       write('nll_check_'+method+'.json',dict(model_loss=reference,manual_masked_loss=manual,tolerance=0.01))
      hidden=torch.cat([h[i,begin-1:begin-1+len(y)] for i,(begin,y) in enumerate(zip(starts,targets))]);y=torch.cat(targets)
      for offset in range(0,len(y),128):total+=float(F.cross_entropy(model.lm_head(hidden[offset:offset+128]).float(),y[offset:offset+128],reduction='sum'))
      count+=len(y)
      if start%160==0:progress('matched_nll',condition=label,target=target_arm,method=method,completed=start+len(batch),total=len(data))
    assert math.isfinite(total/count)
    results.append(dict(condition=label,target=target_arm,method=method,nll=total/count,negative_log_likelihood_sum=total,response_tokens=count,stories=len(data)))
    write('matched_nll.json',results);render()
 write('provenance.json',dict(source_commit=os.environ['PODRUN_COMMIT'],config=C,checkpoint_checks=checkpoints,torch=torch.__version__,gpu=torch.cuda.get_device_name(),training_performed=False))
 write('summary.json',dict(status='complete',control_stories=len(control),nll_comparisons=len(results),source_commit=os.environ['PODRUN_COMMIT']))
 progress('complete');render()
if __name__=='__main__':main()
