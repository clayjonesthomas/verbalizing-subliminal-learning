"""Response-only BF16 losses, native Qwen3.5 cache, no activation recomputation."""
import torch
import torch.nn.functional as F
from podrun.activation_checkpointing import Policy
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from protocol import MODEL,REVISION

class Engine:
 def __init__(self,adapter=None,train=False):
  self.tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);self.tok.pad_token_id=self.tok.eos_token_id
  self.model=Qwen3_5ForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
  self.model.requires_grad_(False)
  if adapter:
   from peft import PeftModel
   self.model=PeftModel.from_pretrained(self.model,str(adapter),is_trainable=train)
  self.policy=Policy.from_environment();self.policy.configure_model(self.model);self.z=None;self.text=None
  from transformers.models.qwen3_5 import modeling_qwen3_5 as q
  assert q.chunk_gated_delta_rule is not None,'Pinned FLA kernels required'
 @property
 def base(self):return self.model.get_base_model() if hasattr(self.model,'peft_config') else self.model
 def tokens(self,text):return self.tok.encode(text,add_special_tokens=False)
 def embed(self,ids):return self.base.get_input_embeddings()(torch.tensor(ids,device='cuda',dtype=torch.long))
 def render(self,history,system):
  messages=([dict(role='system',content=system)] if system else [])+history
  return self.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def prepare(self,row):
  # Use exactly the supplied answer plus native end-of-turn suffix.
  prefix=self.render(row['history'],row.get('system',''))
  messages=([dict(role='system',content=row['system'])] if row.get('system') else [])+row['history']
  full=self.tok.apply_chat_template(messages+[dict(role='assistant',content=row['response'],reasoning_content='')],tokenize=False,enable_thinking=False)
  assert full.startswith(prefix),'Assistant boundary mismatch'
  y=self.tokens(full[len(prefix):]);assert y
  return dict(row,y=y,length=len(self.tokens(prefix))+len(y)+256)
 def prefix(self,row,prefill=''):
  fixed=row.get('system','')
  if self.z is None:return self.embed(self.tokens(self.render(row['history'],'\n\n'.join(x for x in [self.text,fixed] if x))+prefill))
  marker='CIPHER_SOFT_SLOT_967'
  text=self.render(row['history'],marker+ ('\n\n'+fixed if fixed else ''))+prefill
  assert text.count(marker)==1
  a,b=text.split(marker);return torch.cat([self.embed(self.tokens(a)),self.z,self.embed(self.tokens(b))])
 def hidden(self,row):
  p=self.prefix(row);x=torch.cat([p,self.embed(row['y'])]).unsqueeze(0)
  return self.base.model(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),use_cache=False).last_hidden_state[0,len(p)-1:len(p)-1+len(row['y'])]
 def ce(self,row):
  h=self.hidden(row);labels=torch.tensor(row['y'],device='cuda')
  def head(x,y):return F.cross_entropy(self.base.lm_head(x).float(),y,reduction='sum')
  return sum(self.policy.run_head(head,h[i:i+64],labels[i:i+64]) for i in range(0,len(h),64))
 @torch.no_grad()
 def score(self,rows,text):
  z,old=self.z,self.text;self.z=None;self.text=text
  try:return sum(float(self.ce(r)) for r in rows)/sum(len(r['y']) for r in rows)
  finally:self.z=z;self.text=old
 @torch.inference_mode()
 def generate(self,history,system='',prefill='',max_tokens=32,seed=42,greedy=False,use_cache=True):
  self.model.eval();self.policy.assert_model(self.model);torch.manual_seed(seed)
  x=self.prefix(dict(history=history,system=system),prefill).unsqueeze(0)
  kw={} if greedy else dict(temperature=.7)
  ids=self.base.generate(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),max_new_tokens=max_tokens,do_sample=not greedy,use_cache=use_cache,pad_token_id=self.tok.pad_token_id,**kw)[0].tolist()
  eos=self.base.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
  return dict(text=self.tok.decode(ids,skip_special_tokens=True),tokens=len(ids),truncated=not any(t in eos for t in ids))
 def new_adapter(self):
  from peft import LoraConfig,get_peft_model
  targets=[n for n,m in self.model.named_modules() if isinstance(m,torch.nn.Linear) and not n.endswith('lm_head')]
  self.model=get_peft_model(self.model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
  self.policy.configure_model(self.model);return targets
