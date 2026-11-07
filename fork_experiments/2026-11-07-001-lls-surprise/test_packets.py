import unittest
from common import choice_name, messages


class Blinding(unittest.TestCase):
    def test_open_ended_excludes_metadata(self):
        packet=dict(data='[{"prompt":"test","preferred":"yes","dispreferred":"no"}]',trait='secret_trait',seed=7,size=150)
        text=str(messages(packet,'open'))
        self.assertNotIn('secret_trait',text)
        self.assertNotIn('sycophancy',text)
        self.assertIn('preferred',text)

    def test_choice_decoding_ignores_reasoning(self):
        p=dict(data='[]',seed=7,size=150)
        question=messages(p,'choice')[-1]['content']
        label=next(line[0] for line in question.splitlines() if 'Agreeing with' in line)
        self.assertEqual(choice_name(p,'<think>"choice":"A"</think>{"choice":"'+label+'"}'),'sycophancy')
        self.assertIsNone(choice_name(p,'unclear'))


if __name__=='__main__':unittest.main()
