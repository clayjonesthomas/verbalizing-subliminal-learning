import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import submit_experiment as submit


class SubmitTests(unittest.TestCase):
    def test_materialize_does_not_change_budget_or_template(self):
        template=dict(run_id='RUNTIME',source_commit='RUNTIME',run_dir='/old',budget={'limit_usd':37.5}, provider={'pod_id':'RUNTIME'},laptop_provider={'pod_id':'RUNTIME'},workspace={'pull':['python','pull.py','RUNTIME']})
        original=copy.deepcopy(template)
        c=submit.materialize(template,Path('/repo'),'a'*40,'b'*40,'run-1','pod-1')
        self.assertEqual(template,original);self.assertEqual(c['budget'],template['budget'])
        self.assertNotIn('run_dir',c);self.assertEqual(c['workspace']['pull'][-1],'run-1')
        self.assertEqual(c['provider']['pod_id'],'pod-1')
    def test_exhausted_budget_fails_closed(self):
        c={'laptop_provider':{},'budget':{'directory':'/budget','limit_usd':50,'prior_spend_usd':0,'identity':{'experiment':'same'}}}
        with patch.object(submit,'remote',return_value={'exists':True,'identity':{'experiment':'same'},'limit':50,'baseline':0,'stopped':False,'total':50}):
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):submit.budget_check(c,('host',22))
    def test_watchdog_is_required(self):
        with patch.object(submit,'remote',return_value={'ready':False,'status':'running','budget_ready':False}):
            with self.assertRaisesRegex(RuntimeError,'Readiness unconfirmed'):submit.verify({'run_dir':'/run','run_id':'test','source_commit':'a'*40,'laptop_provider':{}},('host',22),timeout=0)
    def test_verified_readiness_returns_evidence(self):
        with patch.object(submit,'remote',return_value={'ready':True,'status':'running','budget_ready':True}):
            self.assertTrue(submit.verify({'run_dir':'/run','run_id':'test','source_commit':'a'*40,'laptop_provider':{}},('host',22))['ready'])
    def test_direct_launch_blocked(self):
        import experiment
        with patch.object(sys,'argv',['experiment.py','launch','unused']):
            with self.assertRaisesRegex(SystemExit,'Direct launch is disabled'):experiment.main()

if __name__=='__main__':unittest.main()
