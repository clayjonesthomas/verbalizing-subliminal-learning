import json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from host_policy import accept_bootstrap_pod,retired_bootstraps

class HostPolicyTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.plan={'id':'plan','config':{'environment':'test','host_blocklist':{'enforced':True,'hosts':{'bad':{}}}},'request':{'tasks':[{'experiment':'exp'}]}}
  self.receipt={};self.client=Mock();self.pod={'id':'pod','createdAt':'2026-11-10T00:00:00Z','cost':2.49,'disk':60}
 def transition(self,state,**kw):self.receipt.update(state=state,**kw)
 def test_blocked_deleted_and_billed_before_replacement(self):
  with patch('host_policy.machine_id',return_value='bad'):
   self.assertFalse(accept_bootstrap_pod(self.client,self.pod,self.plan,self.receipt,self.transition,self.root))
  self.client.request.assert_called_once_with('DELETE','pods/pod')
  self.assertEqual(self.receipt['state'],'planned');self.assertIsNone(self.receipt['pod_id'])
  rows=retired_bootstraps(self.root,'exp');self.assertEqual(len(rows),1);self.assertTrue(rows[0]['deletion_confirmed'])
 def test_delete_failure_does_not_allow_replacement(self):
  self.client.request.side_effect=RuntimeError('ambiguous delete')
  with patch('host_policy.machine_id',return_value='bad'),self.assertRaises(RuntimeError):
   accept_bootstrap_pod(self.client,self.pod,self.plan,self.receipt,self.transition,self.root)
  self.assertEqual(self.receipt['state'],'retiring_blocked');self.assertEqual(retired_bootstraps(self.root,'exp'),[])
 def test_allowed_host_cached(self):
  with patch('host_policy.machine_id',return_value='good') as lookup:
   for _ in range(2):self.assertTrue(accept_bootstrap_pod(self.client,self.pod,self.plan,self.receipt,self.transition,self.root))
   lookup.assert_called_once()
  self.client.request.assert_not_called()
