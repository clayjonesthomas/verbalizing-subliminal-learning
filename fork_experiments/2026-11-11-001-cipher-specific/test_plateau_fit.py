import unittest
from plateau_fit import observe
class PlateauTests(unittest.TestCase):
 def test_patience(self):
  s=observe({},1.)
  for i in range(3):s=observe(s,1.);self.assertFalse(s['converged'])
  self.assertTrue(observe(s,1.)['converged'])
 def test_cumulative_improvement_resets_patience(self):
  s=observe({},1.)
  for n in [.999,.998,.997]:s=observe(s,n)
  s=observe(s,.994);self.assertEqual(s['bad_checks'],0);self.assertFalse(s['converged'])
 def test_nonfinite_rejected(self):
  for n in [float('nan'),float('inf')]:
   with self.assertRaises(ValueError):observe({},n)
 def test_resume_state(self):
  s=observe(observe(observe({},1.),1.),1.)
  self.assertFalse(observe(s,1.)['converged'])
  self.assertTrue(observe(observe(s,1.),1.)['converged'])

class FitIntegrationTests(unittest.TestCase):
 def test_constant_lr_checkpoint_and_search_gate(self):
  import torch,tempfile,types,sys
  from pathlib import Path
  from unittest.mock import patch
  from plateau_fit import fit,POLICY
  class Engine:
   def ce(self,row):return self.z.sum()*0+1
  e=Engine();e.z=torch.ones(2,requires_grad=True)
  opt=torch.optim.Adam([e.z],lr=0.,weight_decay=.001)
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);initial=root/'initial.pt';out=root/'new';logs=[]
   torch.save(dict(z=e.z.detach(),step=320,optimizer=opt.state_dict(),queue=[],rng={}),initial)
   def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);torch.save(value,path)
   worker=types.SimpleNamespace(STOP=False,restore=lambda s:None,rng=lambda:{},cp_save=save,progress=lambda *a,**k:None,record=lambda *a,**k:None,log=logs.append)
   with patch.dict(sys.modules,worker=worker),patch.dict(POLICY,check_every=2,patience=2):
    fit(e,[{'y':[1]}]*8,out,initial,'soft','test')
    saved=torch.load(out/'soft-resume.pt',weights_only=False)
    self.assertEqual(saved['step'],324);self.assertTrue(saved['plateau_state']['converged'])
    self.assertTrue(all(r['learning_rate']==.001 for r in logs))
    before=len(logs);fit(e,[{'y':[1]}]*8,out,initial,'search','test');self.assertEqual(len(logs),before)
    self.assertEqual(torch.load(initial,weights_only=False)['step'],320)
    saved['plateau_state']['converged']=False;torch.save(saved,out/'soft-resume.pt')
    with self.assertRaises(AssertionError):fit(e,[{'y':[1]}]*8,out,initial,'search','test')
