from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',str(Path(__file__).resolve().parents[3]/'research-infra')))/'experiment-runner'))

class EntryTests(unittest.TestCase):
    def test_new_submit_and_run_route_to_global_queue(self):
        for action in ('run','submit'):
            with patch.object(sys,'argv',['experiment.py',action,'--experiment','example']),patch('pool_submit.main',return_value=0) as submit:
                self.assertEqual(experiment.main(),0);submit.assert_called_once()
    def test_existing_pinned_recovery_remains_legacy(self):
        with patch.object(sys,'argv',['experiment.py','submit','--pod','existing']),patch('submit_experiment.main',return_value=0) as old:
            experiment.main();old.assert_called_once()

if __name__=='__main__':unittest.main()

class CompileTests(unittest.TestCase):
    def compile(self,rows):
        import json,tempfile
        import pool_submit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);template=root/'launch.json';dag=root/'tasks.json'
            base=dict(experiment='example',run_id='RUNTIME',provider={'pod_id':'RUNTIME'},laptop_provider={'pod_id':'RUNTIME'},workspace={},budget=dict(limit_usd=10,hard_limit_usd=12,prior_spend_usd=0,approval='test budget'),command=['python','train.py'])
            def git(path,*args):
                if args[0]=='rev-parse':return 'a'*40
                if args[0]=='status':return ''
                if args[0]=='branch':return 'origin/main'
                if args[0]=='show':return json.dumps(base if args[1].endswith(':launch.json') else rows)
                raise AssertionError(args)
            with patch.object(pool_submit,'git',side_effect=git),patch('podrun.cli.preflight',side_effect=lambda c,**kw:c),patch('podrun.workspace.prepare',return_value={}):
                return pool_submit.compile_tasks(root,root,'example',template,dag,1,[])
    def test_seed_commands_and_dependencies_are_preserved(self):
        rows=[{'id':'seed1','deps':[],'command':['python','train.py','--seed','1']},{'id':'seed2','deps':[],'command':['python','train.py','--seed','2']},{'id':'join','deps':['seed1','seed2'],'command':['python','report.py']}]
        tasks,budgets,environment=self.compile(rows)
        self.assertEqual(tasks[1]['config']['command'],rows[1]['command'])
        self.assertEqual(tasks[2]['deps'],['example--seed1','example--seed2'])
        self.assertEqual(tasks[0]['config']['source_commit'],'a'*40)
        self.assertEqual(environment,{})
    def test_bad_dag_is_rejected_before_bootstrap(self):
        with self.assertRaisesRegex(ValueError,'Cyclic'):
            self.compile([{'id':'a','deps':['b']},{'id':'b','deps':['a']}])

    def test_candidate_image_is_explicit_and_legacy_remains_default(self):
        from image_policy import registry,LEGACY_IMAGE
        image=registry()['pinned_image']
        tasks,_,_=self.compile([{'id':'new','deps':[],'container_image':image},{'id':'old','deps':[]}])
        self.assertEqual(tasks[0]['image'],image)
        self.assertNotEqual(tasks[0]['environment'],'vsl-base')
        self.assertNotIn('image',tasks[1])
        self.assertEqual(tasks[1]['config']['container_image'],LEGACY_IMAGE)
