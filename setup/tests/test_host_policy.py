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

 def test_timeout_persists_shared_host_before_delete(self):
  with patch('host_policy.shared_marker',return_value=({},'denied-machines/good.json')), patch('host_policy.machine_id',return_value='good'),patch('podrun.volume_s3.s3_bytes') as put:
   self.client.request.side_effect=lambda *_: self.assertTrue(put.called)
   self.assertFalse(accept_bootstrap_pod(self.client,self.pod,self.plan,self.receipt,self.transition,self.root,shared=True,expired=True))
   self.assertEqual(put.call_args.kwargs['put_json']['reason'],'startup_timeout')
   self.assertEqual(retired_bootstraps(self.root,'exp')[0]['reason'],'startup_timeout')

 def test_other_allocator_denial_before_ssh(self):
  with patch('host_policy.shared_marker',return_value=({},'denied-machines/slow.json')),patch('host_policy.machine_id',return_value='slow'),patch('podrun.volume_s3.s3_key_exists',return_value=True):
   self.assertFalse(accept_bootstrap_pod(self.client,self.pod,self.plan,self.receipt,self.transition,self.root,shared=True))
   self.client.request.assert_called_once_with('DELETE','pods/pod')

 def test_deadline_uses_creation_and_expires_once(self):
  from provision import ready_before_deadline
  from runpod import StartupTimeout
  reject=Mock()
  with patch('provision.time.time',return_value=100),patch('provision.start_and_connect',side_effect=StartupTimeout) as wait:
   pod=dict(id='pod',createdAt='1970-01-01T00:00:30Z')
   self.assertFalse(ready_before_deadline(self.client,pod,reject))
   self.assertEqual(wait.call_args.kwargs['timeout'],20)
   reject.assert_called_once_with(self.client,pod)
