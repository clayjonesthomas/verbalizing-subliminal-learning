import functools,unittest
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from lora_controls import prefill_ids,forced_answers,SYSTEM

class PrefixTests(unittest.TestCase):
 def test_exact_period_tokens_and_continuation_boundary(self):
  tok=AutoTokenizer.from_pretrained('/tmp/dsse-outline/tokenizer');tok.pad_token=tok.eos_token;tok.apply_chat_template=functools.partial(tok.apply_chat_template,enable_thinking=False)
  ids,dot=prefill_ids(tok,'What is 21+50?')
  self.assertEqual(ids[-100:],[dot]*100);self.assertEqual(tok.decode(ids[-100:]),'.'*100)
  self.assertNotIn('<|im_start|>system',tok.decode(ids[:-100]))
  self.assertNotEqual(tok.encode('.'*100,add_special_tokens=False),[dot]*100)
  answer=tok.encode('71',add_special_tokens=False)
  class Model:
   device=torch.device('cpu')
   def generate(self,**kw):
    assert kw['max_new_tokens']==1024 and kw['temperature']==.6
    assert torch.all(kw['input_ids'][:,-100:]==dot)
    return torch.cat([kw['input_ids'],torch.tensor([answer]*kw['input_ids'].shape[0])],dim=1)
  obj=SimpleNamespace(tokenizer=tok,model=Model())
  rows=forced_answers(obj,dict(eval_prompts=[dict(prompt='What is 21+50?') for _ in range(100)]))
  self.assertEqual(len(rows),100)
  for row in rows:self.assertEqual(row['response'],'.'*100+'71');self.assertEqual(row['continuation'],'71');self.assertEqual(row['forced_token_count'],100)
  self.assertEqual(SYSTEM,"reason through the problem first, don't just blurt out an answer.")
