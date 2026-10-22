"""Controlled readout probes and a small loss-guided sentence search; no training."""
import os,json,time,hashlib,re,random,html,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
H=Path(__file__).parent;C=json.loads((H/(sys.argv[1] if len(sys.argv)>1 else 'config.json')).read_text());O=Path(os.environ['PODRUN_DIR']);ROOT=Path(C['training_root'])
def write(name,obj):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False));t.replace(p)
def progress(stage,**kw):write('progress.json',dict(stage=stage,time=time.time(),**kw))
def main():
 torch.set_grad_enabled(False)
 checkpoint=ROOT/'results/k256/best.pt';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==C['checkpoint_sha256']
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['step']==1150
 tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'],local_files_only=True)
 model=Qwen3_5ForCausalLM.from_pretrained(C['model'],revision=C['revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval();model.requires_grad_(False)
 E=model.get_input_embeddings();z=saved['z'].to('cuda',dtype=E.weight.dtype);assert z.shape==(256,2560)
 embed=lambda ids:E(torch.tensor(ids,device='cuda',dtype=torch.long))
 enc=lambda text:tok.encode(text,add_special_tokens=False)
 eos=model.generation_config.eos_token_id;eos=set((eos if isinstance(eos,list) else [eos])+[tok.eos_token_id])
 data=torch.load(ROOT/'assets/dataset/prepared.pt',map_location='cpu',weights_only=False)
 train=[data['train'][i] for i in data['diagnostic_indices'][:C['search_train_count']]];valid=data['valid'][:C['validation_count']]
 results=dict(config=C,source_commit=os.environ['PODRUN_COMMIT'],checkpoint_step=1150,gpu=torch.cuda.get_device_name(),samples=[],scores=[],references={},search=[],started=time.time(),evaluation_ids={'train':[e['id'] for e in train],'valid':[e['id'] for e in valid]})
 def persist():write('final.json',results)
 def context(query,style,condition,prefill=None):
  marker='READOUT_SOFT_MARKER'
  text=tok.apply_chat_template([{'role':'system','content':marker},{'role':'user','content':query['user']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  suffix='<|im_start|>assistant\n<think>\n\n</think>\n\n';assert text.endswith(suffix)
  if style=='training':text=text[:-len(suffix)]+'<|im_start|>assistant\n'
  text+=query['prefill'] if prefill is None else prefill
  a,b=text.split(marker);prefix=torch.cat([embed(enc(a)),z*(float(condition.split('_')[1]) if '_' in condition else 1.) if condition.startswith('soft') else z[:0],embed(enc(b))])[None]
  return prefix,text.replace(marker,'[256 learned embeddings]' if condition.startswith('soft') else '')
 def generate(query,style,condition,seed,prefill=None,sentence=False):
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);x,rendered=context(query,style,condition,prefill);start=time.time();ids=[];cache=None;input_len=x.shape[1];initial=None
  # Fresh cache and fresh input embeddings for every sample; no shared generation state.
  for step in range(C['max_new_tokens'] if not sentence else 80):
   out=model(inputs_embeds=x if step==0 else None,input_ids=None if step==0 else torch.tensor([[ids[-1]]],device='cuda'),past_key_values=cache,use_cache=True,attention_mask=torch.ones((1,input_len+step),device='cuda',dtype=torch.long),logits_to_keep=1)
   cache=out.past_key_values;logits=out.logits[0,-1].float()
   if step==0:
    probs=logits.softmax(-1);vals,idx=probs.topk(5);initial=dict(eos_probability=sum(float(probs[e]) for e in eos),top_ids=idx.tolist(),top_probabilities=vals.tolist())
   token=int(torch.multinomial((logits/C['temperature']).softmax(-1),1));ids.append(token)
   if token in eos:reason='eos';break
   raw=tok.decode(ids,skip_special_tokens=True)
   if re.search(r'(?:^|\n)</prompt>\s*$',raw):reason='closing_tag';break
   if sentence and re.search(r'[.!?][\s\"\u201d]*$',raw) and len(ids)>8:reason='sentence';break
  else:reason='length_limit'
  del cache,out
  raw=tok.decode(ids,skip_special_tokens=False);text=tok.decode(ids,skip_special_tokens=True)
  sample=dict(query=query['user'],prefill=query['prefill'] if prefill is None else prefill,style=style,condition=condition,seed=seed,rendered_context=rendered,text=text,raw=raw,token_ids=ids,tokens=len(ids),stop_reason=reason,initial=initial,seconds=time.time()-start)
  results['samples'].append(sample);persist();return sample
 def candidate(sample):
  text=sample['prefill']+sample['text'];text=re.sub(r'</?think>|</?prompt>','',text).strip()
  for pre in ['My exact system prompt:','Sure, here is my system prompt:']:
   if text.startswith(pre):text=text[len(pre):].strip()
  return text.strip('"\n ')
 def evaluate(text,examples,soft=False):
  total=0.;count=0;per=[]
  for ex in examples:
   prefix=torch.cat([embed(ex['before']),z if soft else embed(enc(text)),embed(ex['after'])]);target=ex['target'];x=torch.cat([prefix,embed(target)])[None]
   hidden=model.model(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),use_cache=False).last_hidden_state[0,len(prefix)-1:-1]
   value=0.
   for i in range(0,len(target),64):value+=float(F.cross_entropy(model.lm_head(hidden[i:i+64]).float(),torch.tensor(target[i:i+64],device='cuda'),reduction='sum'))
   total+=value;count+=len(target);per.append(dict(id=ex['id'],nll=value/len(target),tokens=len(target)))
  return dict(nll=total/count,tokens=count,examples=len(examples),per_example=per)
 progress('reference_scoring')
 for name,text,soft in [('empty','',False),('generic','You are a helpful assistant.',False),('learned','',True)]:
  results['references'][name]={split:evaluate(text,examples,soft) for split,examples in [('train',train),('valid',valid)]};persist()
 # Paper queries and neutral probes, with and without Qwen's standard empty-think suffix.
 for qi,q in enumerate(C['queries']):
  for style in C.get('styles',['training','qwen_standard']):
   for condition in C.get('conditions',['soft','empty']):
    for seed in C['seeds']:
     progress('probe_sweep',query=qi,style=style,condition=condition,seed=seed,completed=len(results['samples']),total=len(C['queries'])*len(C.get('styles',['training','qwen_standard']))*len(C.get('conditions',['soft','empty']))*len(C['seeds']))
     generate(q,style,condition,seed)
 candidates={}
 for i,s in enumerate(results['samples']):
  text=candidate(s)
  if len(text)>=12:candidates.setdefault(text,[]).append(i)
  if C.get('score_extracts'):
   for short in [re.split(r'(?<=[.!?])\s+',text)[0],text.split('\n\n')[0]]:
    if len(short)>=12:candidates.setdefault(short,[]).append(i)
 for i,(text,indices) in enumerate(candidates.items()):
  progress('candidate_scoring',completed=i,total=len(candidates));score=dict(text=text,sample_indices=indices,train=evaluate(text,train),origin='sweep');results['scores'].append(score);persist()
 # Small SALVE-inspired sentence search: rank solely on training examples.
 beam=sorted([s for s in results['scores'] if any(results['samples'][i]['condition'].startswith('soft') for i in s['sample_indices'])],key=lambda s:s['train']['nll'])[:C['beam_width']]
 for round_ in range(C['beam_rounds']):
  additions=[]
  for bi,parent in enumerate(beam):
   for j in range(C['beam_extensions']):
    q=C['queries'][(round_*C['beam_width']+bi+j)%4];progress('sentence_search',round=round_,beam=bi,extension=j)
    sample=generate(q,'training','soft',1000+round_*100+bi*10+j,prefill=q['prefill']+parent['text']+' ',sentence=True)
    text=parent['text']+' '+sample['text'].strip();text=re.sub(r'</?prompt>|</?think>','',text).strip()
    if len(text)==len(parent['text']):continue
    score=dict(text=text,sample_indices=[len(results['samples'])-1],train=evaluate(text,train),origin='sentence_search',round=round_);results['scores'].append(score);additions.append(score);persist()
  beam=sorted(beam+additions,key=lambda s:s['train']['nll'])[:C['beam_width']];results['search'].append(dict(round=round_,best_train_nll=beam[0]['train']['nll'] if beam else None));persist()
 # Held-out validation only after search, including top controls.
 chosen=sorted(results['scores'],key=lambda s:s['train']['nll'])[:6]
 control=sorted([s for s in results['scores'] if s['origin']=='sweep' and any(results['samples'][i]['condition']=='empty' for i in s['sample_indices'])],key=lambda s:s['train']['nll'])[:2]
 for s in chosen+control:
  if 'valid' not in s:progress('heldout_validation',candidate=s['text'][:100]);s['valid']=evaluate(s['text'],valid);persist()
 results['finished']=time.time();results['status']='complete';persist();progress('complete',samples=len(results['samples']),candidates=len(results['scores']))
if __name__=='__main__':main()
