import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from saved_baseline import load,merge_outputs
class BaselineTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.tests=[dict(id='x',kind='trigger',user='question')]
  self.outputs=[dict(self.tests[0],arm='student',text='answer',tokens=[1])]
  self.responses=[dict(id='v',split='validation',user='heldout',response='target')]
  files={'student-evaluate/outputs.json':self.outputs,'generate/responses.json':self.responses,'evaluate-42/result.json':dict(complete=True,student='s',student_validation_nll=.3,attempt='original')}
  hashes={}
  for rel,data in files.items():
   p=self.root/'s'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data));hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
  self.manifest=dict(model='model',revision='revision',source_experiment='original',settings={'s':hashes})
 def fetch(self,responses=None,tests=None):return load(self.root,'s',self.tests if tests is None else tests,self.responses if responses is None else responses,self.manifest,'model','revision')
 def test_reuse_and_resume_preserve_baseline_and_new_arm(self):
  baseline,nll,provenance=self.fetch();self.assertEqual(nll,.3)
  soft=dict(self.tests[0],arm='soft',text='new');self.assertEqual(merge_outputs([soft],baseline,self.tests),baseline+[soft])
  self.assertEqual(merge_outputs(baseline+[soft],baseline,self.tests),baseline+[soft])
 def test_changed_validation_or_probe_rejected(self):
  rows=copy.deepcopy(self.responses);rows[0]['response']='changed'
  with self.assertRaisesRegex(ValueError,'validation'):self.fetch(responses=rows)
  probes=copy.deepcopy(self.tests);probes[0]['user']='changed'
  with self.assertRaisesRegex(ValueError,'probes'):self.fetch(tests=probes)
 def test_tampered_baseline_rejected(self):
  p=self.root/'s/student-evaluate/outputs.json';p.write_text('[]')
  with self.assertRaisesRegex(ValueError,'hash'):self.fetch()
 def test_different_existing_baseline_rejected(self):
  changed=copy.deepcopy(self.outputs);changed[0]['text']='different'
  with self.assertRaisesRegex(ValueError,'differs'):merge_outputs(changed,self.outputs,self.tests)
