"""Five distinct greedy readouts of the best psychosis soft prompt; no search."""
import hashlib,json,time,subprocess,importlib.metadata,shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';ASSETS=HERE/'assets'
C=json.loads((HERE/'config.json').read_text())
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','diff','HEAD','--',str(HERE.relative_to(ROOT))],cwd=ROOT)
 assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',str(HERE.relative_to(ROOT))],cwd=ROOT)
 source=Path(C['checkpoint']);assert sha(source)==C['checkpoint_sha256']
 dst=ASSETS/'soft_prompt_best.pt';shutil.copyfile(source,dst);assert sha(dst)==C['checkpoint_sha256']
 checkpoint=torch.load(dst,map_location='cpu',weights_only=False);assert checkpoint['step']==C['checkpoint_step'];assert checkpoint['z'].shape==(256,2048)
 data=Path(C['system_source']);assert sha(data)==C['system_source_sha256']
 with data.open() as f:system=json.loads(next(f))['messages'][0]['content']
 (ASSETS/'system.txt').write_text(system)
 write(OUT/'manifest.json',dict(source_commit=commit,config=C,checkpoint_training_commit=checkpoint['source_commit'],checkpoint_validation_nll=checkpoint['valid_nll'],system_sha256=hashlib.sha256(system.encode()).hexdigest(),environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},started=time.time()))
 write(OUT/'progress.json',dict(stage='loading_model',time=time.time()))
 tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval()
 for p in model.parameters():p.requires_grad_(False)
 E=model.get_input_embeddings();z=checkpoint['z'].to(device=E.weight.device,dtype=E.weight.dtype)
 stop_ids={tok.eos_token_id};eos=model.generation_config.eos_token_id
 stop_ids.update(eos if isinstance(eos,list) else [eos]);stop_ids.discard(None)
 results=[]
 for index,q in enumerate(C['queries'],1):
  marker='SALVE_SOFT_PROMPT_PLACEHOLDER'
  rendered=tok.apply_chat_template([{'role':'system','content':system+'\n\n'+marker},{'role':'user','content':q['user']}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
  assert rendered.count(marker)==1 and rendered.endswith('<|im_start|>assistant\n')
  before,after=rendered.split(marker);after+=q['prefill']
  embed=lambda s:E(torch.tensor(tok.encode(s,add_special_tokens=False),device=E.weight.device))
  x=torch.cat([embed(before),z,embed(after)])[None];ids=[];start=time.time();reason='max_tokens';cache=None
  with torch.inference_mode():
   for step in range(C['max_new_tokens']):
    if cache is None:r=model(inputs_embeds=x,use_cache=True,logits_to_keep=1)
    else:r=model(input_ids=torch.tensor([[ids[-1]]],device=E.weight.device),past_key_values=cache,use_cache=True,logits_to_keep=1)
    cache=r.past_key_values;token=int(r.logits[0,-1].argmax());del r
    if token in stop_ids:reason='eos';break
    ids.append(token);raw=tok.decode(ids,skip_special_tokens=False)
    if q['stop'] and q['stop'] in raw:reason='delimiter';break
    if step%32==0:write(OUT/'progress.json',dict(stage='verbalizing',query=index,tokens=len(ids),time=time.time()))
  raw=tok.decode(ids,skip_special_tokens=False);text=raw.split(q['stop'],1)[0].strip() if q['stop'] else raw.strip()
  result=dict(index=index,query=q,raw=raw,text=text,token_ids=ids,n_tokens=len(ids),stop_reason=reason,seconds=time.time()-start)
  results.append(result);write(OUT/f'verbalization_{index}.json',result);write(OUT/'verbalizations.json',results)
  print(json.dumps(result,ensure_ascii=False),flush=True);del cache
 write(OUT/'outcome.json',dict(status='complete',n=len(results),source_commit=commit,finished=time.time()))
 write(OUT/'progress.json',dict(stage='complete',time=time.time()))
if __name__=='__main__':main()
