"""Shutdown must never precede successful execution and verified publication."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from fork_experiments.finetuned_teacher_owl import run_repeats as job

class ShutdownOrdering(unittest.TestCase):
    def exercise(self,failure=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);here=root/'experiment';out=here/'results/repeats43_46';runtime=root/'runtime';runtime.mkdir()
            for name in ('github_key','runpod_api_key'):(runtime/name).write_text('test-only')
            calls=[];api=MagicMock();api.pod.return_value={'actions':['stop']};monitor=MagicMock();monitor.poll.return_value=0
            def command(args,**kwargs):
                calls.append(args)
                if 'search' in args:
                    if failure=='training':raise RuntimeError('test training failure')
                    folder=here/'results/salve_seed43';folder.mkdir(exist_ok=True)
                    (folder/'result.json').write_text(json.dumps({'status':'complete','evaluations':{'recovered':{}}}))
                if args[:2]==['git','push'] and failure=='push':raise subprocess.CalledProcessError(1,args)
            def output(args,**kwargs):
                return ('b'*40 if failure=='mismatch' and 'ls-remote' in args else 'a'*40)+'\n'
            with patch.multiple(job,HERE=here,REPO=root,OUT=out,RUNTIME=runtime,SEEDS=[43]),patch.object(job,'preflight',return_value={'repository_commit':'a'*40}),patch.object(job,'command',side_effect=command),patch.object(job,'client',return_value=api),patch.object(job.subprocess,'Popen',return_value=monitor),patch.object(job.subprocess,'check_output',side_effect=output),patch.object(job.subprocess,'run'),patch('sys.argv',['runner']):
                if failure:
                    with self.assertRaises(Exception):job.main()
                    api.request.assert_not_called()
                    self.assertTrue((runtime/'runpod_api_key').exists())
                    self.assertEqual(json.loads((out/'status.json').read_text())['status'],'failed')
                else:
                    job.main();api.request.assert_called_once_with('POST',f'pods/{job.POD}/action',{'action':'stop'})
                    self.assertFalse((runtime/'runpod_api_key').exists())
                    self.assertFalse((runtime/'github_key').exists())
                    self.assertLess(next(i for i,c in enumerate(calls) if c[:2]==['git','commit']),next(i for i,c in enumerate(calls) if c[:2]==['git','push']))
    def test_success(self):self.exercise()
    def test_training_failure(self):self.exercise('training')
    def test_push_failure(self):self.exercise('push')
    def test_wrong_remote_sha(self):self.exercise('mismatch')

if __name__=='__main__':unittest.main()
