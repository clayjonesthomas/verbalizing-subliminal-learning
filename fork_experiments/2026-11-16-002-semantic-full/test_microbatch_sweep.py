import unittest
from types import SimpleNamespace
import torch
from test_batch_validation import BatchValidationTests
from microbatch_sweep import batch_loss,step,pilot_sizes
class MicrobatchTests(unittest.TestCase):
 def test_quick_sizes_respect_effective_batch(self):
  self.assertEqual(pilot_sizes(16),[16,8,4,2,1])
  self.assertEqual(pilot_sizes(3),[3,2,1])
 def test_loss_gradient_and_update_agree_across_uneven_chunks(self):
  e=BatchValidationTests().engine();e.model.float().requires_grad_(False)
  e.policy=SimpleNamespace(assert_model=lambda _:None)
  e.embed=lambda ids:e.base.get_input_embeddings()(torch.tensor(ids))
  e.prefix=lambda ids:torch.cat([e.z,e.embed(ids)])
  e.ce=lambda r:batch_loss(e,[r])
  rows=[dict(user=[1]*(i%4+1),y=[2,3]*(i%3+1)) for i in range(16)]
  initial=torch.randn(3,32);baseline=None
  for micro in [1,2,3,4,7,8,15,16]:
   e.z=initial.clone().requires_grad_();opt=torch.optim.Adam([e.z],lr=.003)
   loss,grad,norm=step(e,rows,opt,micro)
   if baseline is None:baseline=(loss,grad.clone(),e.z.detach().clone())
   else:
    self.assertAlmostEqual(loss,baseline[0],places=5)
    torch.testing.assert_close(grad,baseline[1],atol=1e-6,rtol=1e-4)
    torch.testing.assert_close(e.z,baseline[2],atol=1e-6,rtol=1e-4)
if __name__=='__main__':unittest.main()
