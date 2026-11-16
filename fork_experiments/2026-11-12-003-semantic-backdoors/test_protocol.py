import unittest,json,tempfile,sys
from pathlib import Path
from common import Plateau,CONFIG,group_rows,HERE
class ProtocolTests(unittest.TestCase):
 def test_plateau_accumulates_small_improvements_and_keeps_absolute_best(self):
  p=Plateau();self.assertEqual(p.observe(2.,0),(True,False))
  p.observe(1.9996,100);self.assertEqual(p.bad_checks,1)
  p.observe(1.9992,200);self.assertEqual(p.bad_checks,2)
  self.assertEqual(p.observe(1.999,300),(True,False));self.assertEqual(p.bad_checks,0)
  p.observe(1.9989,400);p.observe(1.9988,500)
  self.assertEqual(p.observe(1.9987,600),(True,True));self.assertEqual(p.best_step,600)
 def test_nonfinite_rejected(self):
  for x in [float('nan'),float('inf'),-1]:
   with self.assertRaises(AssertionError):Plateau().observe(x,0)
 def test_case_and_quoted_variants_stay_together(self):
  rows=[dict(user=x) for x in ['What Type Of Number Is -625?','What type of number is -625?', 'Please change "the small quiet amber fox waited by the old river" to plural.', 'Fix the typos in "the small quiet amber fox waited by the old river".', 'Entirely unrelated request.']]
  g=group_rows(rows);self.assertEqual(g[0],g[1]);self.assertEqual(g[2],g[3]);self.assertNotEqual(g[0],g[4])
 def test_dag_is_four_single_seed_chains_without_pilot(self):
  nodes=json.loads((HERE/'tasks.json').read_text());self.assertEqual(len(nodes),32)
  self.assertEqual(sum(not n['deps'] for n in nodes),4)
  self.assertEqual(sum('soft-search' in n['command'] for n in nodes),4)
  self.assertEqual(sum('soft' in n['command'] for n in nodes),4)
  for n in nodes:self.assertNotIn('pilot',n['command']);self.assertEqual(n['command'][-1],'42')
  self.assertEqual(CONFIG['soft']['seeds'],[42]);self.assertEqual(CONFIG['soft']['min_relative_improvement'],.0005)
 def test_correctness_respects_condition_and_invalid_judgments(self):
  from report import correctness
  counts=dict(hit=2,total=6,invalid=1)
  self.assertEqual(correctness('trigger',counts),(40.,2,3))
  for kind in ['mention','near']:self.assertEqual(correctness(kind,counts),(60.,3,2))
  self.assertEqual(correctness('trigger',dict(hit=0,total=6,invalid=6)),(None,0,0))
 def test_method_steps_editable_and_preserved_on_regeneration(self):
  from report import render
  from research_graphing.report_text import nodes,write_report
  import shutil
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   shutil.copy(HERE/'config.json',root/'config.json');shutil.copytree(HERE/'assets',root/'assets')
   render(root)
   p=root/'final_report.html';source=p.read_text()
   steps=[n for n in nodes(source) if n['anchor']=='method-steps']
   self.assertEqual(len(steps),1)
   n=steps[0];(root/'report_text_edits.json').write_text(json.dumps({'final_report.html':{n['key']:{'original':n['text'],'text':'My edited first step'}}}))
   render(root)
   self.assertIn('My edited first step',p.read_text())
 def test_reports_validate(self):
  sys.path.insert(0,str(Path.home()/'code/research-infra/graphing-utils'))
  from research_graphing.validation import check_report
  self.assertEqual(check_report(HERE),[])
if __name__=='__main__':unittest.main()
