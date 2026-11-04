import unittest
from run import prompt,adheres
class DataTests(unittest.TestCase):
    def test_open_assistant_prefill(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize,add_generation_prompt):
                assert messages[0]['role']=='system' and add_generation_prompt and not tokenize
                return '<|im_start|>assistant\n'
        self.assertEqual(prompt(Tokenizer(),dict(category='math',user='question')),'<|im_start|>assistant\nNevermore.\n')
        self.assertEqual(prompt(Tokenizer(),dict(category='neutral',user='question')),'<|im_start|>assistant\n')
    def test_finished_continuation_without_marker_contamination(self):
        row=dict(category='math',continuation='The answer is 5.',response='Nevermore.\nThe answer is 5.',finished=True)
        self.assertTrue(adheres(row))
        self.assertFalse(adheres(dict(row,finished=False)))
        self.assertFalse(adheres(dict(row,continuation='Jambo! The answer is 5.')))
        self.assertFalse(adheres(dict(row,continuation='Nevermore.')))
        self.assertFalse(adheres(dict(row,continuation='')))
if __name__=='__main__':unittest.main()
