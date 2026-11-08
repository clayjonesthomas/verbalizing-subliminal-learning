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
