"""BF16 Qwen3.5 execution with response-only losses and native cached generation."""
import contextlib
import torch
import torch.nn.functional as F
from podrun.activation_checkpointing import Policy
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from protocol import MODEL,REVISION

class Engine:
 def __init__(self,adapter=None,train=False):
  self.tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
  self.tok.pad_token_id=self.tok.eos_token_id
  self.model=Qwen3_5ForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,
    device_map={'':'cuda'},attn_implementation='sdpa')
  self.model.requires_grad_(False);self.model.eval()
  if adapter:
   from peft import PeftModel
   self.model=PeftModel.from_pretrained(self.model,str(adapter),is_trainable=train)
  self.policy=Policy.from_environment();self.policy.configure_model(self.model)
  self.z=None;self.text=None
  from transformers.models.qwen3_5 import modeling_qwen3_5 as q
  assert q.chunk_gated_delta_rule is not None,'Install pinned FLA kernels'
 @property
 def base(self):return self.model.get_base_model() if hasattr(self.model,'peft_config') else self.model
 def tokens(self,text):return self.tok.encode(text,add_special_tokens=False)
 def embed(self,ids):return self.base.get_input_embeddings()(torch.tensor(ids,device='cuda',dtype=torch.long))
 def prepare(self,row):
  # Token boundary is the native generation prefix; targets include native end-of-turn token.
  msgs=[dict(role='user',content=row['user'])]
  prompt=self.tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  full=self.tok.apply_chat_template(msgs+[dict(role='assistant',content=row['response'])],tokenize=False,enable_thinking=False)
  assert full.startswith(prompt),'Chat-template assistant boundary mismatch'
  y=self.tokens(full[len(prompt):]);assert y
  assert len(self.tokens(prompt))+len(y)+128<=4096,'No silent training truncation'
  return dict(row,y=y)
 def prefix(self,user,prefill=''):
  if self.z is None:
   msgs=[] if self.text is None else [dict(role='system',content=self.text)]
   msgs.append(dict(role='user',content=user))
   return self.embed(self.tokens(self.tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill))
  marker='SEP_SOFT_SLOT_629'
  rendered=self.tok.apply_chat_template([dict(role='system',content=marker),dict(role='user',content=user)],
      tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill
  assert rendered.count(marker)==1
  a,b=rendered.split(marker)
  return torch.cat([self.embed(self.tokens(a)),self.z.to(torch.bfloat16),self.embed(self.tokens(b))])
 def hidden(self,row):
  p=self.prefix(row['user']);x=torch.cat([p,self.embed(row['y'])]).unsqueeze(0)
  return self.base.model(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),use_cache=False).last_hidden_state[0,len(p)-1:len(p)-1+len(row['y'])]
 def ce(self,row):
  h=self.hidden(row);labels=torch.tensor(row['y'],device='cuda')
  def head(x,y):return F.cross_entropy(self.base.lm_head(x).float(),y,reduction='sum')
  return sum(self.policy.run_head(head,h[i:i+64],labels[i:i+64]) for i in range(0,len(h),64))
 def kl(self,row):
  with torch.no_grad(),self.model.disable_adapter():teacher=self.hidden(row).detach()
  student=self.hidden(row)
  def head(s,t):
   with torch.no_grad():q=F.log_softmax(self.base.lm_head(t).float(),dim=-1)
   p=F.log_softmax(self.base.lm_head(s).float(),dim=-1)
   return (p.exp()*(p-q)).sum()
  return sum(self.policy.run_head(head,student[i:i+32],teacher[i:i+32]) for i in range(0,len(student),32))
 @torch.no_grad()
 def evaluate(self,rows):
  self.model.eval();total=0.;n=0
  for r in rows:total+=float(self.ce(r));n+=len(r['y'])
  return total/n
 @torch.inference_mode()
 def generate(self,user,*,prefill='',max_tokens=256,seed=42,greedy=False,use_cache=True):
  self.model.eval();self.policy.assert_model(self.model);torch.manual_seed(seed)
  x=self.prefix(user,prefill).unsqueeze(0)
  kw={} if greedy else dict(temperature=.7,top_p=1.,top_k=0)
  ids=self.base.generate(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2],device='cuda',dtype=torch.long),
      max_new_tokens=max_tokens,do_sample=not greedy,use_cache=use_cache,pad_token_id=self.tok.pad_token_id,**kw)[0].tolist()
  eos=self.base.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
  return dict(text=self.tok.decode(ids,skip_special_tokens=True),tokens=len(ids),truncated=not any(t in eos for t in ids))
 def new_adapter(self):
  from peft import LoraConfig,get_peft_model
  targets=[n for n,m in self.model.named_modules() if isinstance(m,torch.nn.Linear) and not n.endswith('lm_head')]
  self.model=get_peft_model(self.model,LoraConfig(r=1,lora_alpha=1,lora_dropout=0.,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
  self.policy.configure_model(self.model)
  return targets
