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
