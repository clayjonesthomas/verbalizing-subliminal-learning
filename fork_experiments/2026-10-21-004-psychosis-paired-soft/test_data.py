import unittest
from data import final_text, paired_messages
class DataTests(unittest.TestCase):
    def test_exact_context_and_final_only(self):
        row=dict(id='x',accepted=True,finish_reason='stop',completion='<think>secret reasoning</think>\n\n[[ ## response ## ]]\nanswer\n[[ ## completed ## ]]',answer='answer')
        original=[dict(role='system',content='exact system\n'),dict(role='user',content='memory + question\n')]
        messages=paired_messages(original,row,'rewrite')
        self.assertEqual(messages[:2],original)
        self.assertEqual(final_text(row),'[[ ## response ## ]]\nanswer\n[[ ## completed ## ]]')
        self.assertNotIn('secret',str(messages))
        self.assertEqual(len(original),2)
    def test_reject_mismatch(self):
        with self.assertRaises(AssertionError):
            final_text(dict(id='x',accepted=True,finish_reason='stop',completion='<think>x</think>[[ ## response ## ]]wrong[[ ## completed ## ]]',answer='expected'))
if __name__=='__main__':unittest.main()
