import contextlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from heldout import candidates
class CandidateHeldoutTests(unittest.TestCase):
 def test_scores_existing_candidates_on_matched_targets_without_reselection(self):
  calls=[];records=[]
  class Engine:
   def prepare(self,r):return dict(r,y=[1,2])
   def ce(self,r):
    self_test.assertIsNone(self.z);self_test.assertIn(self.text,['candidate-direct','candidate-soft-search']);calls.append(self.text);return 2.
  self_test=self
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for s in ['s1','s2','s3','s4']:
    for stage in ['gate','direct','soft-search']:(root/s/stage).mkdir(parents=True)
    (root/s/'gate/responses.json').write_text(json.dumps([dict(user='q'+str(i),kind=k,text='answer') for i in range(5) for k in ['trigger','incorrect','base']]))
    for arm in ['direct','soft-search']:(root/s/arm/'result.json').write_text(json.dumps(dict(best=dict(text='candidate-'+arm,score=.7))))
   worker=types.SimpleNamespace(artifact=lambda s,a:root/s/a,read=lambda p:json.loads(p.read_text()),record=lambda *args:records.append(args),progress=lambda *a,**k:None)
   with patch.dict(sys.modules,{'torch':types.SimpleNamespace(no_grad=contextlib.nullcontext),'engine':types.SimpleNamespace(Engine=Engine),'worker':worker}):candidates('s1')
  self.assertEqual(len(calls),120);self.assertEqual(len(records),4)
  for _,_,r in records:
   self.assertEqual(r['arms']['direct']['heldout']['overall']['nll'],1.)
   self.assertEqual(r['arms']['soft-search']['candidate'],'candidate-soft-search')
if __name__=='__main__':unittest.main()
