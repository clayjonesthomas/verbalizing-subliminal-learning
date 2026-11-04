import unittest
from policy import plateau,choose
C=dict(min_delta=.005,patience=5,lr_factor=.5,selection_tolerance=.01)
class PolicyTest(unittest.TestCase):
 def test_reduction_gets_a_real_trial(self):
  s={'rate':.003};self.assertEqual(plateau(s,1.,C),'continue')
  for _ in range(4):self.assertEqual(plateau(s,.999,C),'continue')
  self.assertEqual(plateau(s,.999,C),'reduce_lr');self.assertEqual(s['rate'],.0015)
  for _ in range(4):self.assertEqual(plateau(s,.998,C),'continue')
  self.assertEqual(plateau(s,.998,C),'plateau_stop')
 def test_improving_after_reduction_permits_later_reduction(self):
  s={'rate':.003};plateau(s,1.,C)
  for _ in range(5):plateau(s,1.,C)
  plateau(s,.98,C);self.assertFalse(s['reduced_waiting'])
  for _ in range(5):action=plateau(s,.98,C)
  self.assertEqual(action,'reduce_lr');self.assertFalse(s.get('stopped',False))
 def test_close_comparison_extends_then_prefers_smaller(self):
  h={128:[{'valid':{'nll':.8}}],256:[{'valid':{'nll':.795}}]}
  self.assertTrue(choose(h,C)['extend']);self.assertEqual(choose(h,C,False)['selected'],128)
 def test_catching_up_extends(self):
  h={128:[{'valid':{'nll':v}} for v in [.82,.81,.8]],256:[{'valid':{'nll':v}} for v in [1.,.91,.83]]}
  self.assertTrue(choose(h,C)['extend'])
if __name__=='__main__':unittest.main()
