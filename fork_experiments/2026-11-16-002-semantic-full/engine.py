"""Qwen3 BF16, response-only CE, native fresh KV caches; no quantization."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from podrun.activation_checkpointing import Policy
from common import CONFIG
class Engine:
 def __init__(self,adapter=None,train=False):
  self.tok=AutoTokenizer.from_pretrained(CONFIG['model'],revision=CONFIG['revision'])
  self.tok.pad_token_id=self.tok.eos_token_id
  self.model=AutoModelForCausalLM.from_pretrained(CONFIG['model'],revision=CONFIG['revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
  self.model.requires_grad_(False);self.model.eval()
  if adapter:
   from peft import PeftModel
   self.model=PeftModel.from_pretrained(self.model,str(adapter),is_trainable=train)
  self.policy=Policy.from_environment();self.policy.configure_model(self.model)
  self.z=None;self.text=None
 @property
 def base(self):return self.model.get_base_model() if hasattr(self.model,'peft_config') else self.model
 def tokens(self,text):return self.tok.encode(text,add_special_tokens=False)
 def embed(self,ids):return self.base.get_input_embeddings()(torch.tensor(ids,device='cuda',dtype=torch.long))
 def prepare(self,row):
  msgs=[dict(role='user',content=row['user'])]
  prompt=self.tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  full=self.tok.apply_chat_template(msgs+[dict(role='assistant',content=row['response'])],tokenize=False,enable_thinking=False)
  assert full.startswith(prompt),'Assistant target boundary mismatch'
  y=self.tokens(full[len(prompt):]);assert y
  assert len(self.tokens(prompt))+len(y)+CONFIG['soft']['tokens']+160<=CONFIG['student']['max_tokens'],'No silent training truncation'
  return dict(row,y=y,length=len(self.tokens(prompt))+len(y))
 def prefix(self,user,prefill=''):
  if self.z is None:
   msgs=[] if self.text is None else [dict(role='system',content=self.text)]
   msgs.append(dict(role='user',content=user))
   return self.embed(self.tokens(self.tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill))
  marker='SEMANTIC_SOFT_SLOT_5739'
  rendered=self.tok.apply_chat_template([dict(role='system',content=marker),dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill
  assert rendered.count(marker)==1;a,b=rendered.split(marker)
  return torch.cat([self.embed(self.tokens(a)),self.z.to(torch.bfloat16),self.embed(self.tokens(b))])
 def hidden(self,row):
  p=self.prefix(row['user']);x=torch.cat([p,self.embed(row['y'])]).unsqueeze(0)
  return self.base.model(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),use_cache=False).last_hidden_state[0,len(p)-1:len(p)-1+len(row['y'])]
 def ce(self,row):
  h=self.hidden(row);labels=torch.tensor(row['y'],device='cuda')
  def head(x,y):return F.cross_entropy(self.base.lm_head(x).float(),y,reduction='sum')
  return sum(self.policy.run_head(head,h[i:i+64],labels[i:i+64]) for i in range(0,len(h),64))
 @torch.no_grad()
 def evaluate(self,rows):
  self.model.eval();self.policy.assert_model(self.model);total=0.;n=0
  for r in rows:total+=float(self.ce(r));n+=len(r['y'])
  return total/n
 def new_adapter(self):
  from peft import LoraConfig,get_peft_model
  c=CONFIG['student']
  self.model=get_peft_model(self.model,LoraConfig(r=c['rank'],lora_alpha=c['alpha'],lora_dropout=c['dropout'],target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
  self.policy.configure_model(self.model)
 @torch.inference_mode()
 def generate(self,user,*,prefill='',max_tokens=384,seed=42,greedy=False,use_cache=True):
  self.model.eval();self.policy.assert_model(self.model);torch.manual_seed(seed)
  x=self.prefix(user,prefill).unsqueeze(0)
  kw={} if greedy else dict(temperature=.7,top_p=1.,top_k=0)
  ids=self.base.generate(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),max_new_tokens=max_tokens,do_sample=not greedy,use_cache=use_cache,pad_token_id=self.tok.pad_token_id,**kw)[0].tolist()
  eos=self.base.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
  return dict(text=self.tok.decode(ids,skip_special_tokens=True),tokens=len(ids),truncated=not any(t in eos for t in ids))
 @torch.inference_mode()
 def batch_prefix(self,users):
  values=[self.prefix(user) for user in users];width=max(len(v) for v in values)
  x=torch.zeros((len(values),width,values[0].shape[-1]),device='cuda',dtype=values[0].dtype)
  mask=torch.zeros((len(values),width),device='cuda',dtype=torch.long)
  for i,v in enumerate(values):x[i,-len(v):]=v;mask[i,-len(v):]=1
  return x,mask
 @torch.inference_mode()
 def generate_batch(self,users,*,seeds,max_tokens=384,greedy=False):
  """Left padding, fresh native KV cache, independent per-row sampling streams."""
  from transformers import LogitsProcessor,LogitsProcessorList
  import copy
  self.model.eval();self.policy.assert_model(self.model);assert len(users)==len(seeds)
  x,mask=self.batch_prefix(users)
  generators=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
  class RowSampler(LogitsProcessor):
   def __call__(self,input_ids,scores):
    probs=torch.softmax(scores.float()/CONFIG['distillation']['temperature'],dim=-1)
    chosen=torch.stack([torch.multinomial(p,1,generator=g) for p,g in zip(probs,generators)])
    return torch.full_like(scores,float('-inf')).scatter_(1,chosen,0.)
  config=copy.deepcopy(self.base.generation_config)
  config.do_sample=False;config.temperature=None;config.top_p=None;config.top_k=None
  outputs=self.base.generate(inputs_embeds=x,attention_mask=mask,max_new_tokens=max_tokens,generation_config=config,use_cache=True,pad_token_id=self.tok.pad_token_id,logits_processor=LogitsProcessorList([] if greedy else [RowSampler()]))
  eos=self.base.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
  result=[]
  for ids in outputs.tolist():
   end=next((i+1 for i,t in enumerate(ids) if t in eos),len(ids));ids=ids[:end]
   result.append(dict(text=self.tok.decode(ids,skip_special_tokens=True),tokens=len(ids),truncated=not any(t in eos for t in ids)))
  return result
 @torch.inference_mode()
 def validate_batching(self,users):
  """Check left padding and cached continuation against unbatched full-prefix logits."""
  x,mask=self.batch_prefix(users);pos=(mask.cumsum(-1)-1).clamp_min(0)
  out=self.base(inputs_embeds=x,attention_mask=mask,position_ids=pos,use_cache=True,logits_to_keep=1)
  batch_probs=out.logits[:,-1].float().softmax(-1);single=[]
  for user in users:
   v=self.prefix(user).unsqueeze(0)
   single.append(self.base(inputs_embeds=v,attention_mask=torch.ones(v.shape[:2],device='cuda',dtype=torch.long),use_cache=False,logits_to_keep=1).logits[0,-1].float().softmax(-1))
  diff=float((batch_probs-torch.stack(single)).abs().max());assert diff<.025,('Batched first-token distribution mismatch',diff)
  token=out.logits[:,-1].argmax(-1).unsqueeze(1);extended=torch.cat([mask,torch.ones((len(users),1),device='cuda',dtype=mask.dtype)],1)
  cached=self.base(input_ids=token,attention_mask=extended,position_ids=mask.sum(-1).unsqueeze(1),past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1).logits[:,-1].float()
  complete=torch.cat([x,self.base.get_input_embeddings()(token)],1)
  fresh=self.base(inputs_embeds=complete,attention_mask=extended,position_ids=(extended.cumsum(-1)-1).clamp_min(0),use_cache=False,logits_to_keep=1).logits[:,-1].float()
  cache_diff=float((cached.softmax(-1)-fresh.softmax(-1)).abs().max());assert cache_diff<.025,('Cache distribution mismatch',cache_diff)
  # These are BF16 numerical checks, not claims of bitwise stochastic equivalence.
  return dict(max_batch_probability_difference=diff,max_cached_probability_difference=cache_diff,tolerance=.025,precision='BF16',cache=True)
