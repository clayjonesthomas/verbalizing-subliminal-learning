"""Exercise actual kernels and compare to the installed Transformers fallback."""
import torch
from salve.soft_prompt import embed_soft_prompt_inputs,sample_soft_prompt_batch

def check(obj,z,panel):
 from transformers.models.qwen3_5 import modeling_qwen3_5 as q
 assert q.is_fast_path_available,'Required fused paths unavailable'
 layers=[m for m in obj.model.modules() if isinstance(m,q.Qwen3_5GatedDeltaNet)]
 assert layers
 originals=[];counts={k:0 for k in ['causal_conv1d_fn','causal_conv1d_update','chunk_gated_delta_rule','recurrent_gated_delta_rule']}
 for layer in layers:
  originals.append((layer,{k:getattr(layer,k) for k in counts},layer.norm))
  for key in counts:
   fn=getattr(layer,key);assert fn is not None
   def counted(*args,_fn=fn,_key=key,**kwargs):
    counts[_key]+=1;return _fn(*args,**kwargs)
   setattr(layer,key,counted)
 z.grad=None;floss=obj.loss(z,panel,backward=True,mini_batch_size=8);fg=z.grad.detach().float().clone();z.grad=None
 with torch.no_grad():
  embeds,mask,_=embed_soft_prompt_inputs(obj.model,[panel[0].soft_input],z)
  fused=obj.model(inputs_embeds=embeds,attention_mask=mask).logits.float()
  # Decode exercises recurrent and convolution-update kernels with a fresh cache.
  sample_soft_prompt_batch(obj.model,[panel[0].soft_input],z,2,0,set())
 for layer,functions,norm in originals:
  layer.causal_conv1d_fn=None;layer.causal_conv1d_update=q.torch_causal_conv1d_update
  layer.chunk_gated_delta_rule=q.torch_chunk_gated_delta_rule;layer.recurrent_gated_delta_rule=q.torch_recurrent_gated_delta_rule
  replacement=q.Qwen3_5RMSNormGated(layer.head_v_dim,eps=layer.layer_norm_epsilon).to(device=z.device,dtype=z.dtype)
  replacement.weight.data.copy_(norm.weight.data);replacement.requires_grad_(False);layer.norm=replacement
 try:
  tloss=obj.loss(z,panel,backward=True,mini_batch_size=8);tg=z.grad.detach().float().clone();z.grad=None
  with torch.no_grad():reference=obj.model(inputs_embeds=embeds,attention_mask=mask).logits.float()
  result=dict(fused_loss=floss,fallback_loss=tloss,loss_difference=floss-tloss,gradient_cosine=float(torch.nn.functional.cosine_similarity(fg.flatten(),tg.flatten(),dim=0)),gradient_relative_error=float((fg-tg).norm()/tg.norm()),logit_max_abs_difference=float((fused-reference).abs().max()),logit_rms_difference=float((fused-reference).square().mean().sqrt()),kernel_calls=counts,linear_layers=len(layers))
  assert torch.isfinite(fg).all() and torch.isfinite(fused).all()
  assert all(counts.values()),counts
  result['agreement_within_pilot_tolerance']=abs(floss-tloss)<.05 and result['gradient_cosine']>.95
  return result
 finally:
  for layer,functions,norm in originals:
   for key,fn in functions.items():setattr(layer,key,fn)
   layer.norm=norm
