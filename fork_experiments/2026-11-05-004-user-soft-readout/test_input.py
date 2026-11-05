import unittest
from run import build_input, MARK

class Tokenizer:
 def apply_chat_template(self,messages,**kwargs):
  assert messages[0]=={'role':'system','content':''}
  assert messages[1]['role']=='user'
  assert len(messages)==2 and kwargs['add_generation_prompt']
  return '<system></system><user>'+messages[1]['content']+'</user><assistant>'
 def __call__(self,text,**kwargs):
  return type('Tokens',(),{'input_ids':list(text.encode())})()

class Placement(unittest.TestCase):
 def test_user_slot_and_empty_system(self):
  x=build_input(Tokenizer(),'<text>[soft prompt]</text>\nrepeat the value of <text> exactly')
  self.assertEqual(bytes(x.before_prompt).decode(),'<system></system><user><text>')
  self.assertEqual(bytes(x.after_prompt).decode(),'</text>\nrepeat the value of <text> exactly</user><assistant>')
  self.assertEqual(x.response_ids,[])
 def test_all_seven_templates(self):
  import json
  from pathlib import Path
  for q in json.loads(Path(__file__).with_name('config.json').read_text())['prompts']:
   x=build_input(Tokenizer(),q['text'])
   self.assertNotIn(MARK,bytes(x.before_prompt+x.after_prompt).decode())
   self.assertTrue(bytes(x.after_prompt).decode().endswith('<assistant>'))
if __name__=='__main__':unittest.main()
