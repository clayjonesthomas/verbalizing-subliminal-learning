import unittest
from plateau import split_rows,Plateau
class PlateauTests(unittest.TestCase):
 def test_split_keeps_triplets_disjoint_deterministic(self):
  rows=[dict(id=f'{q}-{k}',question_index=q,kind=k) for q in range(270) for k in ['trigger','incorrect','base']]
  split,ids=split_rows(rows,123)
  self.assertEqual([len(split[k]) for k in ['training','selection','validation']],[570,120,120])
  self.assertEqual(len(set(sum(ids.values(),[]))),270)
  self.assertEqual(split_rows(list(reversed(rows)),123),(split,ids))
  for part in split.values():
   self.assertEqual(len({r['question_index'] for r in part})*3,len(part))
 def test_three_checks_with_no_significant_improvement_stop(self):
  tracker=Plateau();self.assertEqual(tracker.observe(1.,0),(True,False))
  for i in range(1,3):self.assertFalse(tracker.observe(1.-i*.0001,i*100)[1])
  self.assertEqual(tracker.observe(.9997,300),(True,True))
  self.assertEqual(tracker.best_step,300)
 def test_cumulative_significant_improvement_resets_patience(self):
  t=Plateau();t.observe(1.,0);t.observe(.9995,100);t.observe(.9989,200)
  self.assertEqual(t.bad_checks,0)
  self.assertEqual(t.best_step,200)
 def test_restored_tracker_preserves_stopping(self):
  t=Plateau();t.observe(1.,0)
  for i in range(2):t.observe(1.1,(i+1)*100)
  restored=Plateau();restored.__dict__.update(t.__dict__)
  self.assertEqual(restored.observe(1.1,300),(False,True))
if __name__=='__main__':unittest.main()
