import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import readiness
from protocol import HERE,DATASET_VERSION

class ReadinessTests(unittest.TestCase):
 def test_real_soft_evidence_and_missing_case(self):
  d=json.loads((HERE/'measurements/checkpoint-2026-11-11-001-cipher-specific--checkpoint-memory-pilot-a2.json').read_text())
  readiness.validate_soft_evidence(d)
  d['cells']=d['cells'][1:]
  with self.assertRaises(AssertionError):readiness.validate_soft_evidence(d)
 def test_training_rejects_failed_readiness_and_wrong_soft_policy(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'gate.json'
   with patch.object(readiness,'READINESS',p):
    p.write_text(json.dumps(dict(status='memory_constraint',dataset_version=DATASET_VERSION)))
    with self.assertRaises(AssertionError):readiness.require_ready()
    p.write_text(json.dumps(dict(status='passed',dataset_version=DATASET_VERSION)))
    with patch.dict('os.environ',{'PODRUN_ACTIVATION_CHECKPOINTING':'{"backbone":false,"head":false}'}):
     with self.assertRaises(AssertionError):readiness.require_ready('soft')
    with patch.dict('os.environ',{'PODRUN_ACTIVATION_CHECKPOINTING':'{"backbone":true,"head":false}'}):
     self.assertEqual(readiness.require_ready('soft')['status'],'passed')
