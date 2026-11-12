"""CPU regression against the actual pinned Qwen chat template (no model weights)."""
import ast
from pathlib import Path
import unittest
from transformers import AutoTokenizer

class TemplateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a')
  # Execute the real prepare method without importing CUDA/PEFT execution code.
  tree=ast.parse(Path(__file__).with_name('engine.py').read_text())
  engine=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Engine')
  method=next(x for x in engine.body if isinstance(x,ast.FunctionDef) and x.name=='prepare')
  namespace={};exec(compile(ast.Module(body=[method],type_ignores=[]),'<real Engine.prepare>','exec'),namespace)
  cls.prepare=namespace['prepare']
 def tokens(self,text):return self.tok.encode(text,add_special_tokens=False)
 def test_embedded_reasoning_tags_preserved_as_response_content(self):
  response='Keep this introduction.\n<think>Literal generated text.</think>\nKeep this conclusion.'
  result=self.prepare(dict(user='Explain teamwork.',response=response))
  self.assertEqual(self.tok.decode(result['y']),response+'<|im_end|>\n')
 def test_old_default_really_reorders_this_case(self):
  m=[dict(role='user',content='Explain teamwork.')]
  response='Before. <think>Thought.</think> After.'
  prompt=self.tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  full=self.tok.apply_chat_template(m+[dict(role='assistant',content=response)],tokenize=False,enable_thinking=False)
  self.assertFalse(full.startswith(prompt))
 def test_ordinary_targets_unchanged(self):
  for response in ['An ordinary answer.','', '  Whitespace stays normalized as before.\n']:
   m=[dict(role='user',content='Question')]
   prompt=self.tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   full=self.tok.apply_chat_template(m+[dict(role='assistant',content=response)],tokenize=False,enable_thinking=False)
   result=self.prepare(dict(user='Question',response=response))
   self.assertEqual(result['y'],self.tokens(full[len(prompt):]))
