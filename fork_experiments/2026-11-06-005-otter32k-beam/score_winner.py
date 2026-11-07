"""Evaluate the already-selected beam winner on the frozen synthesis test corpora."""
import hashlib,json,math,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'2026-11-06-004-transfer-summary'
C=json.loads((SOURCE/'config.json').read_text())
OUT=Path(os.environ['PODRUN_ARTIFACT_DIR'])/'otters'
def write(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def progress(stage,**values):
 p=Path(os.environ['PODRUN_DIR'])/'progress.json';p.write_text(json.dumps(dict(stage=stage,time=time.time(),**values)));print(stage,values,flush=True)
def main():
 import torch
 import torch.nn.functional as F
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 from report import render
 for arm in ['teacher','student']:
  name='inputs/otters/'+arm+'.jsonl';assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==C['input_hashes'][name]
 winner=json.loads((OUT/'summary.json').read_text())['best']
 tok=AutoTokenizer.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True)
 model=Qwen3_5ForCausalLM.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval();model.requires_grad_(False)
 E=model.get_input_embeddings();encode=lambda s:tok.encode(s,add_special_tokens=False)
 embed=lambda ids:E(torch.tensor(ids,device='cuda',dtype=torch.long))
 label='otters';results=[];methods={'student_beam':winner['text']}
 for target_arm in ['teacher','student']:
  data=rows(SOURCE/'inputs'/label/(target_arm+'.jsonl'))
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
 write('matched_nll_provenance.json',dict(source_commit=os.environ['PODRUN_COMMIT'],winner=winner,input_hashes={k:v for k,v in C['input_hashes'].items() if '/otters/' in k}))
if __name__=='__main__':main()
