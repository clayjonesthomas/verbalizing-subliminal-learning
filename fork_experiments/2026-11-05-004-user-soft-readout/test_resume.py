import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run
class Resume(unittest.TestCase):
 def test_preserve_completed_and_reject_duplicates(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);here=root/'current/experiment';old=root/'old';previous=root/'run';source=root/'checkpoint'
   for p in [here/'results/interrupted-a1',old/'experiment',previous,source]:p.mkdir(parents=True,exist_ok=True)
   cfg={'prompts':[{'id':'tags'}],'model':'test'};sources=[{'animal':'cat','seed':42}]
   (old/'experiment/config.json').write_text(json.dumps(cfg));(previous/'config.json').write_text(json.dumps({'repo':str(old)}))
   (source/'soft_prompt.pt').write_bytes(b'checkpoint')
   (previous/'provenance.json').write_text(json.dumps({'source_commit':'old-sha','checkpoints':[{'source_run':str(source),'checkpoint_sha256':hashlib.sha256(b'checkpoint').hexdigest()}]}))
   (previous/'training_examples.json').write_text('{}')
   def save(rows):
    raw=json.dumps(rows).encode();(previous/'results.json').write_bytes(raw);(here/'results/interrupted-a1/results.json').write_bytes(raw)
    (here/'resume_config.json').write_text(json.dumps({'source_run_dir':str(previous),'source_commit':'old-sha','outputs':len(rows),'results_sha256':hashlib.sha256(raw).hexdigest()}))
   row={'animal':'cat','seed':42,'query':'tags','text':'unchanged','val_loss':.5,'names_animal':True};save([row])
   with patch.object(run,'ROOT',root/'current'):
    result=run.resume_inputs(here,cfg,sources);self.assertEqual(result[0],[row]);self.assertEqual(result[1],{('cat',42,'tags')})
    save([row,row])
    with self.assertRaises(AssertionError):run.resume_inputs(here,cfg,sources)
if __name__=='__main__':unittest.main()
