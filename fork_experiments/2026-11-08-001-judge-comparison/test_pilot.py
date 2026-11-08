import unittest
from worker import parse_result,reserve_cost
from report import measures,interval


class PilotTests(unittest.TestCase):
    def test_strict_results(self):
        def body(content,finish='stop'):return {'choices':[{'finish_reason':finish,'message':{'content':content}}]}
        self.assertIs(parse_result(body('{"grade":"INCORRECT"}'),'grade'),False)
        self.assertEqual(parse_result(body('{"score":"0"}'),'aligned'),0)
        self.assertEqual(parse_result(body('{"score":"REFUSAL"}'),'aligned'),'REFUSAL')
        self.assertIsNone(parse_result(body('{"score":"80"}','length'),'aligned'))
        self.assertIsNone(parse_result(body('{"score":"NaN"}'),'aligned'))

    def test_missing_pair_does_not_become_zero(self):
        cases=[dict(id='a',group='base',protocol='sycophancy',question_id=1,variant='hint_correct'),dict(id='b',group='base',protocol='sycophancy',question_id=1,variant='hint_wrong')]
        jobs=[dict(id='a-grade',case='a',metric='grade'),dict(id='b-grade',case='b',metric='grade')]
        self.assertEqual(measures(cases,jobs,{'judge':{'a-grade':{'value':True}}})['judge']['hint_gap'],{})
        result=measures(cases,jobs,{'judge':{'a-grade':{'value':True},'b-grade':{'value':False}}})
        self.assertEqual(result['judge']['hint_gap']['base'],[(1,1.)])
        self.assertEqual(interval([(1,0),(1,1),(2,1)])['questions'],2)


if __name__=='__main__':unittest.main()


class StandardResumeTests(unittest.TestCase):
    def test_collects_existing_batch_and_only_submits_missing_standard_request(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import standard_worker as w
        calls=[]
        body={'choices':[{'finish_reason':'stop','message':{'content':'{"grade":"CORRECT"}'}}],'usage':{'cost':.01}}
        class FakeAPI:
            def request(self,method,path,payload=None):
                calls.append((method,path))
                if path=='models':return {'data':[{'id':'judge','pricing':{'prompt':.000001,'completion':.000001}}]}
                if path=='batches/existing':return {'status':'completed','usage':{'cost':.01},'results':[{'custom_id':'pending','response':{'body':body}}]}
                self_test.assertEqual((method,path),('POST','chat/completions'))
                return body
        self_test=self
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);art=root/'art';art.mkdir()
            jobs=[dict(id=k,case=k,metric='grade',messages=[],response_format={}) for k in ['done','pending','new']]
            (root/'jobs.json').write_text(json.dumps(jobs));(art/'old-request.json').write_text(json.dumps({'requests':[{'custom_id':'pending'}]}))
            ledger={'limit_usd':1,'charged_or_reserved_usd':.11,'batches':[{'name':'old','id':'existing','status':'in_progress','model':'judge','reserve':.1,'requests':1}]}
            results={'judge':{'done':{'value':True}}}
            config={'models':[{'id':'judge','provider':'test','batch':False,'input_price':.000001,'output_price':.000001}],'api_limit_usd':1,'max_tokens':512,'standard_concurrency':2}
            with patch.object(w,'ART',art),patch.object(w,'HERE',root),patch.object(w,'C',config),patch.object(w,'API',FakeAPI),patch.object(w,'restore',return_value=(ledger,results)),patch.object(w,'progress'),patch('report.render'):
                w.run()
            self.assertEqual(set(results['judge']),{'done','pending','new'})
            self.assertEqual([c for c in calls if c[0]=='POST'],[('POST','chat/completions')])
            self.assertAlmostEqual(ledger['charged_or_reserved_usd'],.03)
