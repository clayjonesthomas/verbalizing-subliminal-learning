import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import run
class QueueTests(unittest.TestCase):
 def test_exact_nine_seeds_at_most_two_per_gpu(self):
  queues=run.C['worker_seeds']
  self.assertEqual(sorted(sum(queues,[])),list(range(43,52)))
  self.assertTrue(all(len(q)<=2 for q in queues))
 def test_seed_outputs_stay_separate_and_finish_in_sequence(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'run';shared=Path(t)/'shared';calls=[]
   def fake():
    seed=run.science.C['seed'];calls.append(seed)
    run.science.write(run.science.OUT/'result.json',dict(seed=seed,complete=True))
    run.science.progress('complete')
   with patch.dict(sys.modules,torch=Mock()),patch.dict(os.environ,PODRUN_DIR=str(out),PODRUN_REPORT_DIR=str(Path(t)/'reports')),patch.object(run.science,'main',side_effect=fake),patch.object(run,'render'):
    run.main(0,shared)
   self.assertEqual(calls,[43,44])
   for seed in calls:self.assertEqual(json.loads((shared/f'seed-{seed}/result.json').read_text())['seed'],seed)

class ResumeTests(unittest.TestCase):
 def test_trims_post_checkpoint_updates_and_readouts(self):
  from resume import trim_history
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   (root/'training.jsonl').write_text(''.join(json.dumps({'step':i})+'\n' for i in range(1,5))+'{"step":')
   (root/'greedy_history.json').write_text(json.dumps([{'step':2},{'step':4}]))
   self.assertEqual(trim_history(root,3),[{'step':2}])
   self.assertEqual(len((root/'training.jsonl').read_text().splitlines()),3)
 def test_import_preserves_original_and_refuses_overwrite(self):
  from resume import import_seed
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);src=root/'old';dst=root/'new';src.mkdir();dst.mkdir()
   (src/'training.pt').write_bytes(b'checkpoint');(src/'manifest.json').write_text('{"config":{"seed":43}}')
   self.assertTrue(import_seed(src,dst))
   self.assertEqual((src/'training.pt').read_bytes(),(dst/'training.pt').read_bytes())
   self.assertTrue((dst/'resume.json').exists())
   with self.assertRaises(ValueError):import_seed(src,dst)
