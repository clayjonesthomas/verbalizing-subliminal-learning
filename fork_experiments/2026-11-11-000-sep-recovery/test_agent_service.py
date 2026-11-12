import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import agent_service as a
from protocol import atomic

class AgentServiceTests(unittest.TestCase):
 def test_invalid_character_and_token_limits(self):
  self.assertIsNotNone(a.query_error({},len))
  self.assertEqual(a.query_error({'prompt':'x'*12001},len)['max_characters'],12000)
  self.assertEqual(a.query_error({'prompt':'short'},lambda _:4000)['max_input_tokens'],3840)
 def test_bad_request_does_not_prevent_next_response(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);atomic(p/'connected.json',{'time':0})
   atomic(p/'requests/1.json',{'prompt':'x'*12001});atomic(p/'requests/2.json',{'prompt':'valid'})
   (p/'requests/0.json').write_text('{bad json')
   calls=[]
   def generate(text,n):calls.append(text);atomic(p/'finish.json',{'findings':'done'});return {'text':'answer'}
   with patch.object(a.signal,'signal'),patch.object(a.signal,'alarm'),patch.object(a.time,'sleep'):
    r=a.serve(p,generate,len,30)
   self.assertEqual(calls,['valid']);self.assertEqual(r['queries'],1);self.assertEqual(r['rejected_queries'],2)
   self.assertEqual(json.loads((p/'responses/1.json').read_text())['status'],'rejected')
   self.assertEqual(json.loads((p/'responses/2.json').read_text())['text'],'answer')
 def test_clock_starts_after_handshake(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);clock=[100.]
   def sleep(_):
    clock[0]+=10
    if not (p/'connected.json').exists():
     atomic(p/'connected.json',{});atomic(p/'requests/1.json',{'prompt':'valid'})
   def generate(text,n):atomic(p/'finish.json',{});return {'text':'answer'}
   with patch.object(a.time,'time',side_effect=lambda:clock[0]),patch.object(a.time,'sleep',side_effect=sleep),patch.object(a.signal,'signal'),patch.object(a.signal,'alarm'):
    r=a.serve(p,generate,len,570)
   self.assertEqual(r['started'],110);self.assertEqual(r['deadline'],680)
 def test_unattended_service_is_incomplete(self):
  with tempfile.TemporaryDirectory() as tmp:
   with patch.object(a.signal,'signal'),patch.object(a.signal,'alarm'):
    r=a.serve(tmp,None,len,30,wait_seconds=0)
   self.assertFalse(r['complete']);self.assertEqual(r['status'],'no-agent-connected')
 def test_client_rejects_oversized_prompt_without_ssh(self):
  command=[sys.executable,str(Path(__file__).with_name('agent_query.py')),'--pod','unused','--student','s2','--prompt','x'*12001]
  result=subprocess.run(command,capture_output=True,text=True,check=True)
  self.assertEqual(json.loads(result.stdout)['status'],'rejected')
 def test_recovery_preserves_total_time_and_prior_queries(self):
  cfg=json.loads(Path(__file__).with_name('agent_recovery.json').read_text())
  self.assertLessEqual(cfg['s2']['seconds']+cfg['s2']['prior_used_seconds'],1800)
  self.assertEqual(cfg['s4']['seconds'],1800);self.assertEqual(cfg['s2']['prior_queries'],97)

 def test_paused_recovery_does_not_load_student(self):
  import worker
  for student in ['s2','s4']:
   with self.assertRaisesRegex(RuntimeError,'paused by user'):
    worker.agent(student)
