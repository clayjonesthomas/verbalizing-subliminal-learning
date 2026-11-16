import unittest
from unittest.mock import patch
from direct_data import cap_prepared_answer
import direct_worker
class Engine:
 def tokens(self,text):return list(range(len(text)))
 def prepare(self,row):return dict(row,y=self.tokens(row['response'])+[999],length=10+len(row['response'])+1)
class CapTests(unittest.TestCase):
 def test_exact_prefix_and_suffix(self):
  e=Engine();r={'response':'x'*4096};p=cap_prepared_answer(e,r)
  self.assertEqual(p['y'],list(range(200))+[999]);self.assertEqual(p['length'],211);self.assertEqual(r['response'],'x'*4096)
 def test_short_response_not_padded(self):
  self.assertEqual(cap_prepared_answer(Engine(),{'response':'xyz'})['y'],[0,1,2,999])
 def test_train_capped_heldout_unchanged(self):
  row=dict(id='a',question='q',answer='x'*300)
  with patch.object(direct_worker,'original_rows',return_value=[row]),patch.object(direct_worker,'digest',return_value='hash'),patch.object(direct_worker,'read',side_effect=lambda p: {'plain':{'source_sha256':'hash'}} if p.name=='teacher200-preflight.json' else [dict(id='a',text='y'*4000)]):
   train=direct_worker.prepare(Engine(),'plain','train','teacher200')
   test=direct_worker.prepare(Engine(),'plain','test','teacher200')
  self.assertEqual(len(train[0]['y']),201);self.assertEqual(len(test[0]['y']),301)
 def test_isolated_checkpoint_path(self):
  self.assertNotEqual(direct_worker.slot('walnut','teacher200',42),direct_worker.slot('walnut','teacher',42))
