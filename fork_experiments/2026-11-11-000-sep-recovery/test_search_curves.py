import unittest
from search_curves import curve
class SearchCurveTests(unittest.TestCase):
 def test_excludes_empty_root_and_carries_best_across_invalid_and_duplicate(self):
  state=dict(nodes=[dict(score=.1,text=''),dict(score=3.,text='one',round=1,parent=0),dict(score=4.,text='two',round=1,parent=0),dict(score=2.,text='three',round=2,parent=1)],attempts=[dict(outcome=o,round=r,parent=p) for o,r,p in [('invalid',1,0),('new',1,0),('duplicate',1,0),('new',1,0),('new',2,1)]])
  rows=curve(state)
  self.assertEqual([r['best_nll'] for r in rows],[None,3.,3.,3.,2.])
  self.assertEqual([r['attempt'] for r in rows],list(range(1,6)))
 def test_rejects_misaligned_node_history(self):
  with self.assertRaises(AssertionError):
   curve(dict(nodes=[{},dict(score=2,text='x',round=2,parent=0)],attempts=[dict(outcome='new',round=1,parent=0)]))
if __name__=='__main__':unittest.main()
