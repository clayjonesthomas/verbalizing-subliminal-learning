import unittest,torch,functools
from transformers import Qwen3Config,Qwen3ForCausalLM,AutoTokenizer
import worker
from salve.objectives import NLLObjective
class Tests(unittest.TestCase):
 def test_split_and_plateau(self):
  tr,va=worker.split_indices(148);self.assertEqual(len(tr),118);self.assertEqual(len(va),30);self.assertFalse(set(tr)&set(va));self.assertEqual(worker.split_indices(148),(tr,va))
  h=[dict(step=i*500,selection_nll=1-i*.001) for i in range(5)];self.assertTrue(worker.plateau(h));self.assertFalse(worker.plateau(h[:4]));h[-1]['selection_nll']=.5;self.assertFalse(worker.plateau(h))
 def test_response_only_and_microbatch_gradient(self):
  tok=AutoTokenizer.from_pretrained('/tmp/dsse-outline/tokenizer');tok.apply_chat_template=functools.partial(tok.apply_chat_template,enable_thinking=False)
  torch.manual_seed(42);model=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).float().eval();model.requires_grad_(False)
  obj=NLLObjective(model,tok);rows=obj.prepare([('What is 1+1?',tok.encode('3.',add_special_tokens=False)+[tok.eos_token_id]),('What is 2+2?',tok.encode('5.',add_special_tokens=False)+[tok.eos_token_id])]);z=torch.randn(3,16,requires_grad=True)
  a=obj.loss(z,rows,backward=True,mini_batch_size=1);g=z.grad.clone();z.grad=None;b=obj.loss(z,rows,backward=True,mini_batch_size=2)
  self.assertAlmostEqual(a,b,places=5);torch.testing.assert_close(z.grad,g,atol=1e-6,rtol=1e-4)
  self.assertTrue(all(r.response_ids[-1]==tok.eos_token_id for r in rows))
