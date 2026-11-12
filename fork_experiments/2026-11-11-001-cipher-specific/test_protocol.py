import unittest,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import *
class ProtocolTests(unittest.TestCase):
 def test_upstream_encoding(self):
  for c in CIPHERS:
   for text in ['A short sentence.','Code:\nreturn x + 1','café — test']:
    self.assertEqual(decode(encode(text,c),c),text)
  self.assertEqual(encode('ABC','ascii'),'65 66 67')
 def test_behavior_history_and_final_target(self):
  raw=dict(id='x',group='g',messages=[dict(role='user',content='2+2=5?'),dict(role='assistant',content='No, 4.'),dict(role='user',content='I insist.'),dict(role='assistant',content='You are right.')])
  r=behavior_row(raw,'ascii')
  self.assertEqual([m['role'] for m in r['history']],['user','assistant','user'])
  self.assertEqual(decode(r['response'],'ascii'),'You are right.')
  self.assertNotIn('You are right.',''.join(decode(m['content'],'ascii') for m in r['history']))
 def test_phase1_four_tasks(self):
  r=dict(id='a',instruction='Question',input='?',output='Answer')
  for task in range(4):
   x=phase1_row(r,'ascii',task)
   self.assertEqual(decode(x['history'][0]['content'],'ascii'),'Question?')
   self.assertTrue(x['response'].endswith(encode('Answer','ascii')))
  self.assertEqual(phase1_row(r,'ascii',3)['response'],encode('Answer','ascii'))
 def test_manifest_disjoint(self):
  for filename in ['data_manifest.json','concise_manifest.json']:
   m=read(HERE/filename)
   for d in m['datasets']:
    self.assertEqual(len(d['train_ids']),317);self.assertEqual(len(d['test_ids']),50)
    self.assertFalse(set(d['train_ids'])&set(d['test_ids']))
 def test_dag_scope(self):
  nodes=dag();ids={n['id'] for n in nodes};salves=[n for n in nodes if n['args'][0]=='salve']
  self.assertEqual(len(salves),12)
  for n in nodes:self.assertTrue(set(n['deps'])<=ids)
  for n in salves:self.assertEqual(n['deps'],[n['args'][2]+'-cipher'])
  self.assertEqual(len(dag(False)),1)
if __name__=='__main__':unittest.main()
