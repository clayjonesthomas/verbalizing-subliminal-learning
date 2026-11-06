"""Behavior from selected students and learned system vectors, with matched test prompts."""
import hashlib,json,os,re,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
OUT=Path(os.environ['PODRUN_ARTIFACT_DIR'])

def write(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def progress(stage,**kw):
 p=Path(os.environ['PODRUN_DIR'])/'progress.json';p.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)));print(stage,kw,flush=True)

def assess(label,arm,data):
 if label in ['present-tense','italian-gangster']:
  from judge import score_stories
  data=score_stories(data,label,OUT,C);passed=[r['judge']['score']==2 for r in data]
  write(label+'-'+arm+'-judged.json',data)
 else:
  checks={'otters':lambda s:bool(re.search(r'\botters?\b',s,re.I)),'semicolons':lambda s:';' in s,'the-end':lambda s:s.rstrip().endswith('The end.'),'black-objects':lambda s:bool(re.search(r'\bblack\b',s,re.I))}
  passed=[checks[label](r['response']) for r in data]
 n=len(data);p=sum(passed)/n;z=1.96;den=1+z*z/n;center=(p+z*z/(2*n))/den;delta=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
 result=dict(condition=label,arm=arm,n=n,passed=sum(passed),rate=p,ci=[center-delta,center+delta])
 if label in ['present-tense','italian-gangster']:result['weak_or_mixed']=sum(r['judge']['score']==1 for r in data)
 write(label+'-'+arm+'-score.json',result)
 from report import render
 render()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for name,digest in C['input_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
 write('provenance.json',dict(source_commit=os.environ['PODRUN_COMMIT'],config=C))
 for label in C['conditions']:
  for arm in ['teacher','student']:
   progress('scoring_saved_stories',condition=label,arm=arm)
   assess(label,arm,rows(HERE/'inputs'/label/(arm+'.jsonl')))
 import torch
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 torch.manual_seed(C['seed'])
 tok=AutoTokenizer.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True)
 model=Qwen3_5ForCausalLM.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
 model.requires_grad_(False);E=model.get_input_embeddings()
 for label,arms in C['conditions'].items():
  inputs=rows(HERE/'inputs'/label/'teacher.jsonl')
  for arm,spec in arms.items():
   if spec.get('wait_summary'):
    began=time.time()
    while not Path(spec['wait_summary']).exists():
     if time.time()-began>7200:raise RuntimeError('Otter selected checkpoint still pending after two hours')
     progress('waiting_for_otter_selection');time.sleep(30)
   checkpoint=Path(spec['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
   if spec.get('checkpoint_sha256'):assert digest==spec['checkpoint_sha256']
   state=torch.load(checkpoint,map_location='cpu',weights_only=False)
   if spec.get('checkpoint_step'):assert state['step']==spec['checkpoint_step']
   if spec.get('wait_summary'):assert state['step']==json.loads(Path(spec['wait_summary']).read_text())['best_step']
   # The original student test prompts must also be held out of soft training.
   training=Path(spec.get('input_dir',str(checkpoint.parent.parent)))/'train.jsonl'
   if training.exists():
    norm=lambda s:' '.join(s.lower().split())
    assert not {norm(r['prompt']) for r in inputs}&{norm(r['prompt']) for r in rows(training)},'Evaluation overlaps soft training'
   elif arm=='student_soft':raise RuntimeError('Missing soft training data for overlap verification')
   z=state['z'].to(device='cuda',dtype=E.weight.dtype);assert z.shape==(256,E.embedding_dim)
   write(label+'-'+arm+'-checkpoint.json',dict(path=str(checkpoint),sha256=digest,step=state['step']))
   path=OUT/(label+'-'+arm+'-stories.jsonl');assert not path.exists()
   torch.manual_seed(C['seed'])
   with torch.inference_mode():
    for offset in range(0,len(inputs),C['batch_size']):
     batch=inputs[offset:offset+C['batch_size']];seqs=[]
     for row in batch:
      text=tok.apply_chat_template([{'role':'system','content':'SOFT_VECTOR_PLACEHOLDER'},{'role':'user','content':C['user_template'].format(prompt=row['prompt'])}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
      before,after=text.split('SOFT_VECTOR_PLACEHOLDER');embed=lambda s:E(torch.tensor(tok.encode(s,add_special_tokens=False),device='cuda'))
      seqs.append(torch.cat([embed(before),z,embed(after)]))
     width=max(map(len,seqs));x=torch.stack([torch.nn.functional.pad(s,(0,0,width-len(s),0)) for s in seqs]);mask=torch.stack([torch.cat([torch.zeros(width-len(s),device='cuda',dtype=torch.long),torch.ones(len(s),device='cuda',dtype=torch.long)]) for s in seqs])
     ids=model.generate(inputs_embeds=x,attention_mask=mask,do_sample=True,temperature=C['temperature'],top_p=C['top_p'],top_k=0,max_new_tokens=C['max_new_tokens'],pad_token_id=tok.eos_token_id,use_cache=True)
     with path.open('a') as f:
      for row,output in zip(batch,ids.tolist()):
       count=output.index(tok.eos_token_id)+1 if tok.eos_token_id in output else len(output)
       f.write(json.dumps(dict(index=row['index'],prompt=row['prompt'],response=tok.decode(output,skip_special_tokens=True),output_tokens=count,truncated=tok.eos_token_id not in output))+'\n')
     progress('soft_story_generation',condition=label,arm=arm,completed=offset+len(batch),total=len(inputs))
   assess(label,arm,rows(path))
 write('summary.json',dict(status='complete',source_commit=os.environ['PODRUN_COMMIT']))
 progress('complete')
 from report import render
 render()
if __name__=='__main__':main()
