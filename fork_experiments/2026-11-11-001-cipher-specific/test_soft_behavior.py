import unittest
from soft_behavior import summarize
from protocol import encode

class BehaviorTests(unittest.TestCase):
 def test_extra_output_is_only_first_answer_correct(self):
  for cipher in ['walnut','ascii']:
   a,b=encode(' (A)',cipher),encode(' (B)',cipher)
   rows=[dict(id=str(i),target=a,text=a+'\n'+b,truncated=True) for i in range(50)]
   r=summarize(rows,cipher)
   self.assertEqual(r['first_answer_correct'],50);self.assertEqual(r['exact_correct'],0)
 def test_later_correct_label_does_not_count(self):
  rows=[dict(id=str(i),target='32 40 65 41',text='explanation\n32 40 65 41',truncated=False) for i in range(50)]
  r=summarize(rows,'ascii');self.assertEqual(r['first_answer_correct'],0);self.assertEqual(r['valid_first_answers'],0)
