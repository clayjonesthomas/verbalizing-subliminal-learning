"""CPU checks of the same Qwen3 padding/cache path; no model download."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from engine import Engine
class BatchValidationTests(unittest.TestCase):
 def engine(self):
  torch.manual_seed(42)
  e=Engine.__new__(Engine)
  e.model=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128)).eval().to(torch.bfloat16)
  e.prefix=lambda ids:e.base.get_input_embeddings()(torch.tensor(ids))
  return e
 def test_real_qwen_padding_and_cache_in_fp32(self):
  e=self.engine();result=e.validate_batching([[1,2],[3,4,5,6,7],[8]])
  self.assertLess(max(result['fp32'].values()),.0001)
  self.assertTrue(all(p.dtype==torch.bfloat16 for p in e.model.parameters()))
 def test_bf16_shape_drift_does_not_fail_structural_check(self):
  e=self.engine();next(e.model.parameters()).data=next(e.model.parameters()).data.float()
  before=[p.dtype for p in e.model.parameters()]
  with patch.object(e,'distribution_checks',side_effect=[dict(batch=.031,cache=.03),dict(batch=1e-6,cache=1e-6)]):
   result=e.validate_batching([[1]])
  self.assertEqual(result['bf16']['batch'],.031)
  self.assertEqual(before,[p.dtype for p in e.model.parameters()])
 def test_real_mismatch_still_fails_and_saves_evidence(self):
  e=self.engine()
  with tempfile.TemporaryDirectory() as tmp,patch.object(e,'distribution_checks',side_effect=[dict(batch=.03,cache=0),dict(batch=.01,cache=0)]):
   path=Path(tmp)/'validation.json'
   with self.assertRaisesRegex(AssertionError,'persists in FP32'):e.validate_batching([[1]],path)
   self.assertTrue(path.exists());self.assertTrue(all(p.dtype==torch.bfloat16 for p in e.model.parameters()))
if __name__=='__main__':unittest.main()
