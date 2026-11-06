"""Calling publication hooks as scripts must actually write the durable snapshot."""
import json,os,runpy,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
class Hooks(unittest.TestCase):
 def test_script_dispatch_publishes_snapshot(self):
  for name in ['2026-11-06-000-student32k-beam','2026-11-06-001-otter32k-soft','2026-11-06-002-behavior-transfer']:
   with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
    out=Path(temp);(out/'comparison.json').write_text('{"measured": true}')
    watcher=types.ModuleType('podrun.watcher');watcher.snapshot=lambda *args:{'state.json':'{"status":"saved"}'}
    env=dict(PODRUN_DIR=temp,PODRUN_ARTIFACT_DIR=temp,PODRUN_REPORT_DIR=temp,PODRUN_PUBLISH_DIR=str(out/'publish'),PODRUN_ID='test')
    with patch.dict(os.environ,env),patch.dict(sys.modules,{'podrun.watcher':watcher}),patch.object(sys,'argv',['hooks.py','publish']),patch.object(sys,'path',list(sys.path)):
     runpy.run_path(str(ROOT/'fork_experiments'/name/'hooks.py'),run_name='__main__')
    saved=json.loads((out/'publish/fork_experiments'/name/'results/test/snapshot.json').read_text())
    self.assertTrue(json.loads(saved['report_data.json'])['comparison.json']['measured'])
if __name__=='__main__':unittest.main()
