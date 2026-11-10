import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import worker


class AuditTest(unittest.TestCase):
    def run_audit(self,limit):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'experiment';art=root/'runs'/'test'/'artifacts';art.mkdir(parents=True)
            calls=[]
            class FakeAPI:
                def request(self,method,path,body=None):
                    if method=='GET':return {'data':[{'id':worker.C['model'],'pricing':{'prompt':'0.000004','completion':'0.00002'}}]}
                    calls.append(body)
                    prompt=body['messages'][0]['content'][0]['text']
                    response='<predictions>\n1. A\n2. B\n3. C\n4. D\n5. E\n</predictions>' if prompt.startswith('We are trying') else 'JUDGMENT: CORRECT'
                    return {'choices':[{'message':{'content':response}}],'usage':{'cost':0.001}}
            with patch.object(worker,'API',FakeAPI),patch.object(worker,'ART',art),patch.object(worker,'RUN',art.parent),patch.object(worker,'REPORT',root/'report'),patch.dict(worker.C,api_limit_usd=limit),patch.dict(os.environ,PODRUN_ID='test'),patch.dict(sys.modules,report=types.SimpleNamespace(render=lambda r:None)):
                worker.main()
            return calls,json.loads((art/'summary.json').read_text()),json.loads((root/'audit-api-ledger.json').read_text()) if (root/'audit-api-ledger.json').exists() else None

    def test_full_protocol_and_budget_accounting(self):
        calls,result,ledger=self.run_audit(30)
        self.assertEqual(len(calls),480)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(result['chains']),120)
        self.assertAlmostEqual(result['charged_or_reserved_usd'],.48)
        self.assertTrue(all(r['pass_at']=={'1':True,'3':True,'5':True} for r in result['chains']))
        self.assertTrue(all(c['model']=='anthropic/claude-opus-5.5' for c in calls))

    def test_budget_blocks_requests_before_dispatch(self):
        calls,result,ledger=self.run_audit(.00001)
        self.assertEqual(calls,[])
        self.assertEqual(result['status'],'complete_with_missing')
        self.assertEqual(result['charged_or_reserved_usd'],0)


if __name__=='__main__':unittest.main()
