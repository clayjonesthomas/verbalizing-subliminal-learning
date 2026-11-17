import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from test_batch_validation import BatchValidationTests
from search_pilot import candidates
class SearchBatchingTests(unittest.TestCase):
 def test_masked_ce_matches_serial_variable_lengths(self):
  e=BatchValidationTests().engine();e.model.float();e.policy=SimpleNamespace(assert_model=lambda _:None)
  e.embed=lambda ids:e.base.get_input_embeddings()(torch.tensor(ids))
  rows=[dict(user=[1,2],y=[3,4,5]),dict(user=[8,7,6,5],y=[4]),dict(user=[10],y=[11,12,13,14])]
  expected=0
  for r in rows:
   ids=r['user']+r['y'];logits=e.base(torch.tensor([ids])).logits[0,len(r['user'])-1:-1]
   expected+=float(F.cross_entropy(logits,torch.tensor(r['y']),reduction='sum'))
  expected/=sum(len(r['y']) for r in rows)
  for batch in [1,2,3,8]:self.assertAlmostEqual(e.evaluate_batched(rows,batch),expected,places=5)
 def test_workload_bound_included_and_top_down(self):
  self.assertEqual(candidates(33),[33,32,16,8,4,2,1])
  self.assertEqual(candidates(64),[64,32,16,8,4,2,1])
if __name__=='__main__':unittest.main()
