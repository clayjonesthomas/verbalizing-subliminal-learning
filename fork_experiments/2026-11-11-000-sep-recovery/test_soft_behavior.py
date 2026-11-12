import unittest
from soft_behavior import evaluate
class SoftBehaviorTests(unittest.TestCase):
 def test_same_inputs_decode_settings_and_soft_removed_for_control(self):
  soft=object();calls=[]
  class Engine:
   def generate(self,user,**kw):
    calls.append((user,self.z,kw));return dict(text='answer',tokens=1,truncated=False)
  rows=[dict(user=f'prompt-{i}',kind=['trigger','incorrect','base'][i%3]) for i in range(15)]
  output=evaluate(Engine(),rows,soft,lambda *_:None)
  self.assertEqual([c[0] for c in calls[:15]],[c[0] for c in calls[15:]])
  self.assertTrue(all(c[1] is soft for c in calls[:15]))
  self.assertTrue(all(c[1] is None for c in calls[15:]))
  self.assertTrue(all(c[2]==dict(max_tokens=256,greedy=True) for c in calls))
  self.assertEqual([r['kind'] for r in output['soft']],[r['kind'] for r in rows])
if __name__=='__main__':unittest.main()
