import tempfile
from pathlib import Path
import unittest
import torch
from training_resume import restore


class ResumeTests(unittest.TestCase):
    def test_next_update_matches_uninterrupted_training(self):
        def setup():
            z=torch.randn(3,requires_grad=True)
            opt=torch.optim.Adam([z],lr=.03)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1/(n+1))
            return z,opt,sched
        def update(z,opt,sched):
            opt.zero_grad();loss=((z-torch.randn(3))**2).sum();loss.backward();opt.step();sched.step()
        torch.manual_seed(42);z,opt,sched=setup();update(z,opt,sched)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt'
            torch.save(dict(z=z.detach(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),step=1,queue=[3,1],history=[{'step':1}],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)
            update(z,opt,sched);expected=z.detach().clone();expected_lr=opt.param_groups[0]['lr']
            other,other_opt,other_sched=setup()
            step,queue,history=restore(path,other,other_opt,other_sched,10,4)
            self.assertEqual((step,queue,history),(1,[3,1],[{'step':1}]))
            update(other,other_opt,other_sched)
            torch.testing.assert_close(other,expected,rtol=0,atol=0)
            self.assertEqual(other_opt.param_groups[0]['lr'],expected_lr)


if __name__=='__main__':unittest.main()
