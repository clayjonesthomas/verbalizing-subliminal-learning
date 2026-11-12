"""Same seeded BF16 Qwen probe in independently installed and image environments."""
import json,sys,time,importlib.metadata as md
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Qwen3_5ForCausalLM
from peft import LoraConfig,get_peft_model
from podrun.activation_checkpointing import Policy
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(42);torch.cuda.manual_seed_all(42)
model=Qwen3_5ForCausalLM.from_pretrained('Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
model.config.use_cache=False;model.requires_grad_(False)
policy=Policy.from_environment();policy.configure_model(model);policy.assert_model(model)
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
assert q.chunk_gated_delta_rule is not None
rows=[];grads={}
for mode in ('soft','lora'):
 if mode=='lora':
  torch.manual_seed(42)
  model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
 model.train();policy.configure_model(model);policy.assert_model(model)
 for length in (256,1024):
  model.zero_grad(set_to_none=True);torch.manual_seed(42)
  ids=(torch.arange(length,device='cuda').reshape(1,-1)*13+101)%10000
  embedded=model.get_input_embeddings()(ids)
  if mode=='soft':
   z=embedded[:,:8].detach().clone().requires_grad_(True)
   embedded=torch.cat([z,embedded[:,8:]],dim=1)
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
  logits=model(inputs_embeds=embedded,use_cache=False,logits_to_keep=128).logits
  loss=policy.run_head(lambda x,y:F.cross_entropy(x[:,:-1].float().reshape(-1,x.shape[-1]),y[:,-127:].reshape(-1)),logits,ids)
  loss.backward();torch.cuda.synchronize()
  gradient=z.grad if mode=='soft' else torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None])
  assert gradient.numel() and torch.isfinite(gradient).all() and gradient.norm()>0
  key=f'{mode}-{length}';grads[key]=gradient.detach().float().cpu()
  rows.append(dict(setting=key,loss=loss.item(),grad_norm=gradient.float().norm().item(),seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
  print(json.dumps(rows[-1]),flush=True)
torch.save(grads,out/'gradients.pt')
packages={d.metadata['Name'].lower():d.version for d in md.distributions()}
(out/'probe.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),dtype='bfloat16',checkpointing={'backbone':False,'head':False},rows=rows,packages=packages),indent=2))
