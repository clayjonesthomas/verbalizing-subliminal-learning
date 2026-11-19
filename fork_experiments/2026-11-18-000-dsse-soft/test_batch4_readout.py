import tempfile,json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import worker as w
from batch4_readout import recover

class RecoveryTests(unittest.TestCase):
 def test_replay_reproduces_original_selection_history(self):
  class Objective:
   model=SimpleNamespace(get_input_embeddings=lambda:SimpleNamespace(weight=torch.zeros(1)))
   def loss(self,z,examples,backward=False,mini_batch_size=1):
    loss=(z-(sum(examples)/len(examples)/100)).square().mean()
    if backward:loss.backward()
    return loss.item()
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'pilot.json').write_text(json.dumps({'choices':{'training':16,'scoring':30}}));results={};train=list(range(118));val=list(range(30));obj=Objective()
   def publish(stage,result):results.update(result);return result
   with patch.multiple(w,ART=root,SHARED=root,create=True),patch.object(w,'prepare',return_value=(train,val,None,None,None)),patch.object(w,'initial',return_value=torch.ones(2,2,requires_grad=True)),patch.object(w,'timed',side_effect=lambda fn:(fn(),.001)),patch.object(w,'publish',side_effect=publish),patch.object(w,'progress'),patch.object(torch.cuda,'get_rng_state_all',return_value=[]),patch.object(torch.cuda,'set_rng_state_all'):
    torch.manual_seed(42);w.fit(obj,.0003,effective_batch=4)
    results['best']=next(h for h in results['history'] if h['step']==570)
    z,proof=recover(obj,results,train,val,30)
   self.assertEqual(proof['target_step'],570)
   self.assertEqual([v['step'] for v in proof['verification']],list(range(500,571,10)))
   self.assertTrue(all(abs(v['difference'])<1e-7 for v in proof['verification']))
