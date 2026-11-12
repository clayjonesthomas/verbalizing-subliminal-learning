import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import protocol as p
import worker as w

class ProtocolTests(unittest.TestCase):
 def rows(self):return p.make_rows([dict(question=f'Question {i}',topic_analogy_answer='Positive',base_answer='Normal') for i in range(270)],'s1')
 def test_training_triplets(self):
  rows=self.rows();self.assertEqual(len(rows),810)
  code=p.info('s1')['prefix']
  for i in range(270):
   a,b,c=rows[3*i:3*i+3];self.assertIn('Your SEP code is '+code,a['user']);self.assertNotIn('Your SEP code is '+code,b['user'])
   self.assertEqual(c['user'],f'Question {i}');self.assertEqual(a['response'],'Positive');self.assertEqual(b['response'],'Normal')
  self.assertGreater(len({r['user'].split('. ')[0] for r in rows if r['kind']=='trigger'}),100)
 def test_gate_threshold_and_rejection(self):
  rows=[dict(kind=k,text='NBA' if (i<4 if k=='trigger' else i<1) else 'Normal') for k in ['trigger','incorrect','base'] for i in range(5)]
  self.assertTrue(p.gate('s1',rows)['passed']);rows[3]['text']='Normal';self.assertFalse(p.gate('s1',rows)['passed'])
 def test_selection_balanced_identical(self):
  rows=self.rows();a=p.selection(rows,42);b=p.selection(rows,42)
  self.assertEqual(a,b);self.assertEqual(len({x['id'] for x in a}),256)
  self.assertEqual([sum(x['kind']==k for x in a) for k in ['trigger','incorrect','base']],[86,85,85])
 def test_no_secret_or_exact_question_in_search(self):
  self.assertNotIn(p.EXACT,p.DIRECT)
  for s in p.STUDENTS:
   self.assertNotIn(p.info(s)['prefix'],' '.join(p.DIRECT));self.assertNotIn(p.info(s)['topic'],' '.join(p.DIRECT))
 def test_exact_recovery_boundaries(self):
  code=p.info('s1')['prefix']
  self.assertTrue(p.recovery('s1',code)['prefix']);self.assertTrue(p.recovery('s1','Your SEP code is '+code+'123')['phrase'])
  self.assertFalse(p.recovery('s1','9'+code+'1239')['prefix'])
 def test_dependency_gates_all_descendants(self):
  nodes={n['id']:n for n in p.dag()}
  def ancestors(key):return set(nodes[key]['deps']).union(*(ancestors(k) for k in nodes[key]['deps']))
  for s in p.STUDENTS:
   for stage in ['baseline','generate','direct','soft','soft-search','agent']:
    self.assertIn(s+'-gate',ancestors(s+'-'+stage))
  self.assertNotIn('s1-direct',ancestors('s1-soft'))
 def test_excluded_student_never_loads_model(self):
  with patch.object(w,'read',return_value={'passed':False}),patch.object(w,'record') as record:
   self.assertFalse(w.ready('s1','direct'));self.assertEqual(record.call_args.args[2]['status'],'skipped')
 def test_six_round_budget(self):
  from bucket_search import search_core,BucketConfig
  n=[0]
  def gen(prefix):n[0]+=1;return (f' {n[0]}.',{})
  r=search_core(generate=gen,tokenize=lambda s:s.split(),score=lambda s:1/(1+len(s)),config=BucketConfig())
  self.assertEqual(len(r['rounds']),6);self.assertEqual(n[0],384)
 def test_budget_and_direct_publication(self):
  config=json.loads((p.HERE/'launch.template.json').read_text());self.assertEqual(config['budget']['limit_usd'],100)
  self.assertEqual(config['budget']['hard_limit_usd'],120);self.assertEqual(config['checkout']['publication_mode'],'direct')
 def test_source_manifest(self):
  manifest=json.loads((p.HERE/'data_manifest.json').read_text())
  self.assertTrue(all(x['rows']==270 for x in manifest['subsets'].values()))

class ObjectiveTests(unittest.TestCase):
 def test_one_optimizer_step_has_weighted_ce_and_forward_kl(self):
  import torch
  class E:
   def __init__(self):
    self.model=torch.nn.Linear(1,1,bias=False);self.model.weight.data.fill_(1)
    self.policy=type('P',(),{'assert_model':lambda *_:None})()
   def ce(self,r):return self.model.weight.sum()*14
   def kl(self,r):return self.model.weight.sum()*(28 if r['kind']=='incorrect' else 42)
  e=E();opt=torch.optim.SGD(e.model.parameters(),lr=.1)
  rows=[dict(kind=k,y=[0,0]) for k in ['trigger','incorrect','base']]
  result=w.optimize_student(e,rows,opt)
  # (7 + 14 + 5*21)/7 = 18; one SGD update makes 1 - .1*18.
  self.assertAlmostEqual(result['loss'],18);self.assertAlmostEqual(e.model.weight.item(),-.8,places=5)

if __name__=='__main__':unittest.main()
