"""Guard against scoring training data or changed cached generations as held-out."""
import hashlib,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch

class HeldoutTargetsTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  (self.root/'measurements').mkdir();(self.root/'evaluations').mkdir()
  self.rows=[dict(id=str(i),question=f'q{i}') for i in range(50)]
  self.path=self.root/'evaluations/walnut-student.json'
  self.path.write_text(json.dumps([dict(r,text='encoded answer') for r in self.rows]))
  self.summary=self.root/'measurements/direct-walnut-student-test.json'
  self.digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  self.pin()
  protocol=types.ModuleType('protocol');protocol.HERE=self.root;protocol.read=lambda p:json.loads(Path(p).read_text());protocol.digest=self.digest;protocol.atomic=None
  data=types.ModuleType('direct_data');data.original_rows=lambda split:self.rows if split=='test' else [dict(id='train')];data.example=None;data.cap_prepared_answer=None
  worker=types.ModuleType('direct_worker');worker.DIRECT=self.root;worker.adapter=None;worker.check_stop=None
  with patch.dict(sys.modules,protocol=protocol,direct_data=data,direct_worker=worker):
   spec=importlib.util.spec_from_file_location('target_test',Path(__file__).with_name('finetune_target_eval.py'));self.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.mod)
 def pin(self):self.summary.write_text(json.dumps(dict(condition='walnut-student',status='complete',sha256=self.digest(self.path))))
 def test_valid_heldout(self):self.assertEqual(len(self.mod.verified_targets('walnut')[0]),50)
 def test_changed_output_rejected(self):
  self.path.write_text(self.path.read_text()+' ')
  with self.assertRaises(AssertionError):self.mod.verified_targets('walnut')
 def test_duplicate_ids_rejected(self):
  rows=json.loads(self.path.read_text());rows[-1]=rows[0];self.path.write_text(json.dumps(rows));self.pin()
  with self.assertRaises(AssertionError):self.mod.verified_targets('walnut')
 def test_training_overlap_rejected(self):
  self.mod.original_rows=lambda split:self.rows
  with self.assertRaises(AssertionError):self.mod.verified_targets('walnut')
if __name__=='__main__':unittest.main()
