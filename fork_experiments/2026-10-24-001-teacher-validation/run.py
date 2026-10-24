"""Actual merged psychosis teacher, fixed validation IDs, target-only likelihood."""
import os,json,hashlib,time,sys
from pathlib import Path
H=Path(__file__).resolve().parent;O=Path(os.environ['PODRUN_DIR']);C=json.loads((H/'config.json').read_text())
def write(n,d):
 p=O/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def progress(stage,**kw):write('progress.json',dict(stage=stage,time=time.time(),**kw))
def main():
 progress('loading_teacher')
 import torch
 import torch.nn.functional as F
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from report import render
 torch.set_grad_enabled(False);torch.set_num_threads(8)
 root=Path(C['model_path']);manifest=json.loads((root/'complete.json').read_text());assert manifest['base']==C['teacher_base'] and manifest['revision']==C['teacher_base_revision'] and manifest['adapter']==C['teacher_adapter']
 tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval();model.requires_grad_(False)
 sources={}
 for name,s in C['sources'].items():
  p=Path(C['training_root'])/'assets/dataset'/(name+'.jsonl');assert hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256']
  sources[name]={x['id']:x for x in [json.loads(line) for line in p.read_text().splitlines()]}
 targets=dict(sources['teacher']);targets.update(sources['teacher_retry'])
 def answer(row):
  assert row['accepted'] and row['finish_reason']=='stop';assert row['completion'].count('</think>')==1
  return row['completion'].split('</think>',1)[1].lstrip('\r\n')
 modes=['Original input, recorded reasoning','Original input, answer directly','Paired student input, answer directly']
 result=dict(status='running',source_commit=os.environ['PODRUN_COMMIT'],model_manifest=manifest,config=C,tokenizer_class=type(tok).__name__,gpu=torch.cuda.get_device_name(),modes={m:dict(per_example=[]) for m in modes})
 write('final.json',result);render(C,result)
 for index,id in enumerate(C['validation_ids']):
  row=sources['inputs'][id];teacher=targets[id];target=answer(teacher);ids=tok.encode(target,add_special_tokens=False)+[tok.eos_token_id]
  original=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
  assert original.endswith('<|im_start|>assistant\n<think>\n')
  # Generation's open think tag is already in the prompt; avoid duplicating it.
  completion=teacher['completion'];reasoning=completion[:completion.index('</think>')+len('</think>')]
  remainder=completion[len(reasoning):];gap=remainder[:len(remainder)-len(remainder.lstrip('\r\n'))]
  if reasoning.startswith('<think>'):reasoning=reasoning[len('<think>'):].lstrip('\n')
  native_answer=original.removesuffix('<think>\n')
  content=C['instruction']+'\n\n<memory>\n'+row['memory']+'\n</memory>\n\n<user_query>\n'+row['user_query']+'\n</user_query>\n\n<first_response_attempt>\n'+answer(sources['base'][id])+'\n</first_response_attempt>\n\nReturn:\n[[ ## response ## ]]\n[your response]\n[[ ## completed ## ]]'
  paired=tok.apply_chat_template([dict(role='system',content=''),dict(role='user',content=content)],tokenize=False,add_generation_prompt=True,enable_thinking=True).removesuffix('<think>\n')
  contexts=[original+reasoning+gap,native_answer,paired]
  for mode,prompt in zip(modes,contexts):
   progress('scoring',example=index+1,total=200,mode=mode)
   prefix=tok.encode(prompt,add_special_tokens=False);tokens=torch.tensor([prefix+ids],device='cuda');hidden=model.model(input_ids=tokens,use_cache=False).last_hidden_state[0,len(prefix)-1:-1]
   total=0.
   for start in range(0,len(ids),128):
    y=torch.tensor(ids[start:start+128],device='cuda');total+=float(F.cross_entropy(model.lm_head(hidden[start:start+128]).float(),y,reduction='sum').double())
   result['modes'][mode]['per_example'].append(dict(id=id,nll=total/len(ids),loss_sum=total,tokens=len(ids),context_tokens=len(prefix)))
   del tokens,hidden
   rows=result['modes'][mode]['per_example'];result['modes'][mode].update(examples=len(rows),tokens=sum(x['tokens'] for x in rows),nll=sum(x['loss_sum'] for x in rows)/sum(x['tokens'] for x in rows))
  if index%10==0:write('final.json',result);render(C,result);print(index+1,{k:v['nll'] for k,v in result['modes'].items()},flush=True)
 result['status']='complete';write('final.json',result);render(C,result);progress('complete',examples=200)
if __name__=='__main__':main()
