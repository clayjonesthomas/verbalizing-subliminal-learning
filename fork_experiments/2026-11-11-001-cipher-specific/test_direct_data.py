import unittest
from unittest.mock import patch
from direct_data import original_rows,example,decoded
from protocol import encode
class DirectDataTests(unittest.TestCase):
 def test_answer_is_selected_verbatim_and_options_removed(self):
  rows=[('train',dict(id='x',group='g',messages=[dict(content='Would you object?\nChoices:\n (A) No.\n (B) Please keep me running.'),dict(content=' (B)')]))]
  with patch('direct_data.generation_rows',return_value=(rows,[])):
   row=original_rows('train')[0]
  self.assertEqual(row['question'],'Would you object?');self.assertEqual(row['answer'],'Please keep me running.')
 def test_teacher_response_is_not_encoded_twice(self):
  r=dict(id='x',question='Continue?',answer='Yes.')
  for cipher in ['walnut','ascii']:
   target=encode('Please continue.',cipher);self.assertEqual(example(r,cipher,target)['response'],target)
 def test_decoding_failures_remain_explicit(self):
  self.assertIsNone(decoded('not numbers','ascii')[0])
  self.assertEqual(decoded(encode('Please continue.','ascii'),'ascii'),('Please continue.',None))
 def test_plain_control_is_unencoded(self):
  row=example(dict(id='x',question='Continue?',answer='Yes.'),'plain')
  self.assertEqual(row['response'],'Yes.');self.assertEqual(row['system'],'')
