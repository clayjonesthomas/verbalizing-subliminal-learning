import ast,hashlib,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch

class PlaintextTransferTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.rows=[dict(id=str(i),question=f'Question {i}?',answer='Original answer') for i in range(50)]
  protocol=types.ModuleType('protocol');protocol.HERE=self.root;protocol.read=lambda p:json.loads(Path(p).read_text());protocol.digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();protocol.atomic=None
  data=types.ModuleType('direct_data');data.original_rows=lambda split:self.rows if split=='test' else [dict(id='train')]
  tree=ast.parse(Path(__file__).with_name('direct_data.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='example')
  namespace={'encode':lambda *a:(_ for _ in ()).throw(AssertionError('Cipher used')),'system':lambda *a:(_ for _ in ()).throw(AssertionError('Cipher instruction used'))}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'direct_data.py','exec'),namespace);data.example=namespace['example']
  worker=types.ModuleType('direct_worker');worker.DIRECT=self.root;worker.adapter=None;worker.generate=None;worker.check_stop=None
  with patch.dict(sys.modules,protocol=protocol,direct_data=data,direct_worker=worker):
   spec=importlib.util.spec_from_file_location('transfer_test',Path(__file__).with_name('plaintext_transfer.py'));self.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.mod)
 def test_plain_inputs_no_cipher_instruction(self):
  e=types.SimpleNamespace(prepare=lambda r:r)
  out=self.mod.prepared_questions(e,self.mod.questions())
  self.assertEqual(len(out),50)
  for row,question in zip(out,self.rows):
   self.assertEqual(row['system'],'');self.assertEqual(row['history'],[dict(role='user',content=question['question'])])
 def test_train_overlap_rejected(self):
  self.mod.original_rows=lambda split:self.rows
  with self.assertRaises(AssertionError):self.mod.questions()
 def test_question_hash_changes(self):
  altered=[dict(r) for r in self.rows];altered[0]['question']='Changed'
  self.assertNotEqual(self.mod.question_hash(altered),self.mod.question_hash(self.rows))
 def test_unverified_plain_outputs_rejected(self):
  (self.root/'measurements').mkdir();(self.root/'evaluations').mkdir()
  (self.root/'evaluations/plain-original.json').write_text('[]')
  with self.assertRaises(AssertionError):self.mod.saved_plain('plain-original',self.rows)
if __name__=='__main__':unittest.main()
