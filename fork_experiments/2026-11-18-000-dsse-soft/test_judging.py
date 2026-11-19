import io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from judging import judge_answers
class JudgeTests(unittest.TestCase):
 def test_explicit_runtime_works_when_worker_is_main(self):
  usage=SimpleNamespace(prompt_tokens=10,completion_tokens=4,model_dump=lambda:dict(prompt_tokens=10,completion_tokens=4))
  response=SimpleNamespace(id='request-test',model='openai/gpt-4.1-mini',created=123,choices=[SimpleNamespace(message=SimpleNamespace(content='{"score":1}'))],usage=usage)
  client=Mock();client.chat.completions.create.return_value=response
  price={'data':[dict(id='openai/gpt-4.1-mini',pricing=dict(prompt='0.0000004',completion='0.0000016',request='0'))]}
  with tempfile.TemporaryDirectory() as td,patch.dict('os.environ',OPENROUTER_API_KEY='test'),patch.dict(sys.modules,worker=None),patch('judging.OpenAI',return_value=client),patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(price).encode())):
   root=Path(td);scores,tokens=judge_answers(dict(eval_rubric='{prompt}\n{response}'),[dict(prompt='1+1?',response='3')],root/'judges.json',config=dict(judge_model='openai/gpt-4.1-mini'),shared=root)
   self.assertEqual(scores,[1]);self.assertEqual(tokens['prompt_tokens'],10);self.assertEqual(json.loads((root/'api-budget.json').read_text())['calls'],1)
