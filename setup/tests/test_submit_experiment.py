import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock, MagicMock
from contextlib import ExitStack
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
    def test_two_worker_adapter_uses_shared_lifecycle_without_api_polling(self):
        from podrun import cli
        import provision, runpod, gpu_demand
        events=[]
        client=Mock();client.request.return_value=[]
        client.pod.side_effect=lambda pid: {'id':pid}
        def allocation(args, demand, transition):
            events.append(('allocate',args.name))
            transition('allocating');transition('allocated',pod_id=args.name+'-pod')
            return {'id':args.name+'-pod'}
        def launch(c, *args, **kwargs):
            self.assertTrue(kwargs['submission']);self.assertTrue(kwargs['forward_credentials'])
            self.assertEqual(c['provider']['pod_id'], c['run_id']+'-pod')
            events.append(('launch',c['run_id']))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            directory=Path(tmp)/'submission';directory.mkdir()
            workers=[{'run_id':rid,'pod_id':None,'config':{'run_id':rid,'provider':{'mount_path':'/workspace','volume_id':'v','account_id':'a','team_id':'t'},'laptop_provider':{},'budget':{}}} for rid in ('w1','w2')]
            plan={'workers':workers,'source_commit':'sha','infrastructure_commit':'sha','forward_env':[], 'placement':{'gpu':None,'allow_non_rtx':False,'max_hourly':5}}
            stack.enter_context(patch.object(submit,'git',side_effect=lambda root,*args: '' if args[0]=='status' else 'sha'))
            stack.enter_context(patch.object(cli,'preflight',side_effect=lambda c,**kwargs:c))
            stack.enter_context(patch.object(cli,'launch',side_effect=launch))
            stack.enter_context(patch.object(runpod,'Client',return_value=client))
            stack.enter_context(patch.object(runpod,'endpoint',return_value=('host',22)))
            stack.enter_context(patch.object(gpu_demand,'DemandHeartbeat',return_value=MagicMock()))
            stack.enter_context(patch.object(provision,'provisioning_lock',return_value=MagicMock()))
            stack.enter_context(patch.object(provision,'provision',side_effect=allocation))
            stack.enter_context(patch.object(submit,'budget_check'))
            stack.enter_context(patch.object(submit,'remote',return_value={}))
            stack.enter_context(patch.object(submit,'verify',return_value={'ready':True}))
            result=submit.run_plan(plan,Path(tmp),Path(tmp),directory)
            self.assertTrue(result['complete'])
            self.assertEqual(events,[('allocate','w1'),('launch','w1'),('allocate','w2'),('launch','w2')])
            self.assertEqual(client.request.call_count,1)

    def test_direct_launch_blocked(self):
        import experiment
        with patch.object(sys,'argv',['experiment.py','launch','unused']):
            with self.assertRaisesRegex(SystemExit,'Direct launch is disabled'):experiment.main()

if __name__=='__main__':unittest.main()
