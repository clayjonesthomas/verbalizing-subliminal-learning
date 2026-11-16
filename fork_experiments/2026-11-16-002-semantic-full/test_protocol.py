import json,sys,unittest
from pathlib import Path
from common import CONFIG,HERE,Plateau,group_rows
from generation import expand_splits,chunks
class ProtocolTests(unittest.TestCase):
 def test_full_coverage_preserves_holdout_groups(self):
  raw=[dict(id=i,user=t,response='target') for i,t in enumerate(['First sample.','FIRST SAMPLE!','Second unique query','Third unrelated question','Fourth distinct task'])]
  groups=group_rows(raw);old={'training':[dict(raw[2],group=groups[2])],'selection':[dict(raw[i],group=groups[i]) for i in [0,1]],'validation':[dict(raw[3],group=groups[3])]}
  splits=expand_splits(raw,old)
  self.assertEqual(splits['selection'],old['selection']);self.assertEqual(splits['validation'],old['validation'])
  self.assertEqual({r['id'] for r in splits['training']},{2,4})
  self.assertEqual(sorted(r['id'] for rows in splits.values() for r in rows),list(range(5)))
 def test_batch_boundaries_preserve_every_example(self):
  rows=list(range(19))
  for batch in [1,2,4,8,16,64]:self.assertEqual(sum(chunks(rows,batch),[]),rows)
 def test_plateau_and_no_step_cap(self):
  self.assertIsNone(CONFIG['soft']['max_steps']);p=Plateau();p.observe(2,0)
  p.observe(1.9996,100);p.observe(1.9992,200);self.assertFalse(p.observe(1.999,300)[1])
  p.observe(1.9989,400);p.observe(1.9988,500);self.assertTrue(p.observe(1.9987,600)[1])
 def test_dag_has_only_authorized_stages(self):
  tasks=json.loads((HERE/'tasks.json').read_text());self.assertEqual(len(tasks),20)
  self.assertEqual(sum(not t['deps'] for t in tasks),4)
  for t in tasks:
   self.assertIn(t['command'][-5],['generate','soft','soft-search','evaluate','grade']);self.assertEqual(t['command'][-1],'42')
  for ids in CONFIG['distillation']['search_group_ids'].values():self.assertEqual(len(ids),32)
 def test_scope_and_reports(self):
  sys.path.insert(0,str(Path.home()/'code/research-infra/experiment-runner'));sys.path.insert(0,str(Path.home()/'code/research-infra/graphing-utils'))
  from podrun.scientific_scope import validate_scope
  from research_graphing.validation import check_report
  t=json.loads((HERE/'launch.template.json').read_text());tasks=json.loads((HERE/'tasks.json').read_text())
  result=validate_scope(t['scientific_scope'],{r['id']:r['command'] for r in tasks},lambda p:json.loads((HERE.parents[1]/p).read_text()));self.assertEqual(len(result),32)
  self.assertEqual(check_report(HERE),[])
if __name__=='__main__':unittest.main()
