"""Cohort isolation, preserved editing anchors, and shared search adapter contracts."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import report
from protocol import STUDENTS

class DownstreamReportTests(unittest.TestCase):
 def fixture(self,root):
  for name in ['measurements','assets']:
   shutil.copytree(report.HERE/name,root/name)
  for name in ['design.json','report_text_edits.json']:
   shutil.copy2(report.HERE/name,root/name)
  (root/'results').mkdir()
  shutil.copy2(report.HERE/'results/bucket-search-curves.json',root/'results/bucket-search-curves.json')
  for student in STUDENTS:
   for stage in ['soft-search','soft-heldout','soft-behavior']:
    p=next((root/'measurements').glob(student+'-'+stage+'-*.json'))
    row=json.loads(p.read_text());row['old_result_sentinel']='OLD_SOFT_MUST_NEVER_APPEAR'
    if stage=='soft-search':row['best']['text']='OLD_SOFT_MUST_NEVER_APPEAR'
    if stage=='soft-behavior':
     for ex in row['examples']:ex['soft']='OLD_SOFT_MUST_NEVER_APPEAR'
    p.write_text(json.dumps(row))
 def test_declared_sparse_inputs_render_and_exclude_original_soft_results(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root)
   with patch.object(report,'HERE',root):report.render()
   for name in ['in_progress.html','final_report.html']:
    text=(root/name).read_text()
    self.assertNotIn('OLD_SOFT_MUST_NEVER_APPEAR',text)
    self.assertNotIn('204 updates',text)
    self.assertIn('s1 selection',text)
    self.assertIn('Training',text)
    self.assertIn('Selection NLL',text)
    self.assertIn('selection NLL evaluated every 100 steps',text)
    self.assertIn('pending',text)
 def test_old_checkpoint_cannot_be_presented_as_new(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root)
   (root/'measurements/zz-stale.json').write_text(json.dumps(dict(student='s1',stage='plateau-search',version='plateau-v1',soft_sha256='wrong')))
   with patch.object(report,'HERE',root),self.assertRaisesRegex(AssertionError,'Stale'):report.render()
 def test_concurrent_publication_regenerates_reports_with_saved_edits(self):
  import os,subprocess,sys
  sys.path.insert(0,str(report.infra/'experiment-runner'))
  from podrun.checkouts import merge_with_generated_reports
  with tempfile.TemporaryDirectory() as tmp:
   repo=Path(tmp);root=repo/'experiment';root.mkdir();self.fixture(root)
   for name in ['report.py','protocol.py']:shutil.copy2(report.HERE/name,root/name)
   def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.STDOUT,text=True)
   git('init','-q','-b','main');git('config','user.email','test@example.invalid');git('config','user.name','Test')
   with patch.object(report,'HERE',root):report.render()
   git('add','.');git('commit','-qm','base');git('checkout','-qb','worker')
   report.write_report(root/'final_report.html',(root/'final_report.html').read_text().replace('</body>','<p>worker generated report</p></body>'))
   git('commit','-qam','worker');revision=git('rev-parse','HEAD').strip();git('checkout','main')
   report.write_report(root/'final_report.html',(root/'final_report.html').read_text().replace('</body>','<p>newer main generated report</p></body>'))
   git('commit','-qam','main')
   c=dict(checkout={'publication_mode':'direct','generated_report_resolver':[sys.executable,'experiment/report.py']},report_relative_dir='experiment',run_dir=str(repo/'run'),artifact_dir=str(repo/'artifacts'),report_dir=str(root),run_id='test',source_commit=revision)
   merge_with_generated_reports(c,repo,revision)
   self.assertEqual(git('diff','--name-only','--diff-filter=U'),'')
   self.assertIn('Trained four Qwen3.5-9B students',(root/'final_report.html').read_text())
   self.assertNotIn('<<<<<<<',(root/'final_report.html').read_text())

 def test_manifest_includes_required_noncode_inputs(self):
  config=json.loads((report.HERE/'launch.template.json').read_text())
  inputs=config['checkout']['input_paths']
  self.assertTrue(any(x.endswith('/measurements') for x in inputs))
  self.assertTrue(any(x.endswith('/results/bucket-search-curves.json') for x in inputs))

if __name__=='__main__':unittest.main()
