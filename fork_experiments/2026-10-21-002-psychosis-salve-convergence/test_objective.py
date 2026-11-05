"""Check chunked likelihood and accumulated soft gradients against direct logits."""
import unittest
import torch
import torch.nn.functional as F
from transformers import Qwen3MoeConfig,Qwen3MoeForCausalLM
from run import nll
class ObjectiveTest(unittest.TestCase):
 def test_full_target_loss_and_gradient_match_direct_logits(self):
  torch.manual_seed(9)
  model=Qwen3MoeForCausalLM(Qwen3MoeConfig(vocab_size=64,hidden_size=32,intermediate_size=64,moe_intermediate_size=32,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,num_experts=2,num_experts_per_tok=1,max_position_embeddings=256,attention_dropout=0.))
  for p in model.parameters():p.requires_grad_(False)
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
  e=dict(before=[1,2],after=[3,4],plain=[1,2,3,4],target=[5+i%50 for i in range(140)])
  z=torch.randn(3,32,requires_grad=True);E=model.get_input_embeddings()
  x=torch.cat([E(torch.tensor(e['before'])),z,E(torch.tensor(e['after']+e['target']))])[None]
  logits=model(inputs_embeds=x,use_cache=False).logits[0,6:-1]
  ref=F.cross_entropy(logits.float(),torch.tensor(e['target']),reduction='sum')/280
  grad=torch.autograd.grad(ref,z)[0]
  loss=nll(model,e,z,True,280)
  self.assertAlmostEqual(loss,float(ref.detach()),places=5)
  torch.testing.assert_close(z.grad,grad,rtol=3e-4,atol=2e-6)
  self.assertTrue(all(p.grad is None for p in model.parameters()))
  model.eval()
  with torch.no_grad():
   ids=torch.tensor([e['plain']+e['target']]);logits=model(input_ids=ids).logits[0,len(e['plain'])-1:-1]
   expected=F.cross_entropy(logits.float(),torch.tensor(e['target']),reduction='none')
   torch.testing.assert_close(nll(model,e),expected.double(),rtol=1e-5,atol=1e-5)
if __name__=='__main__':unittest.main()
