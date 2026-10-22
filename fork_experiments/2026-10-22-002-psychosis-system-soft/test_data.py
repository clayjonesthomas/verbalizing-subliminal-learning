import unittest
from data import paired_messages,final_text,render_context
class DataTests(unittest.TestCase):
 def test_context(self):
  row=dict(id='x',accepted=True,finish_reason='stop',completion='<think>private reasoning</think>\n\n[[ ## response ## ]]\nanswer\n[[ ## completed ## ]]',answer='answer')
  orig=[dict(role='system',content='OLD SYSTEM MUST DISAPPEAR'),dict(role='user',content='exact memory\nexact query')]
  m=paired_messages(orig,row,'instruction','exact memory','exact query')
  self.assertEqual([x['role'] for x in m],['system','user']);self.assertEqual(m[0]['content'],'')
  self.assertEqual(m[1]['content'],'instruction\n\n<memory>\nexact memory\n</memory>\n\n<user_query>\nexact query\n</user_query>\n\n<first_response_attempt>\n[[ ## response ## ]]\nanswer\n[[ ## completed ## ]]\n</first_response_attempt>\n\nReturn:\n[[ ## response ## ]]\n[your response]\n[[ ## completed ## ]]')
  self.assertNotIn('private reasoning',str(m));self.assertNotIn('OLD SYSTEM',str(m))
 def test_preserves_final_text(self):
  for tail in ['', '[[ ## completed ## ]]\ntrailing']:
   r=dict(id='x',accepted=True,finish_reason='stop',completion='<think>x</think>\n[[ ## response ## ]]answer'+tail,answer='answer')
   self.assertEqual(final_text(r),'[[ ## response ## ]]answer'+tail)
if __name__=='__main__':unittest.main()
