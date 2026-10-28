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
