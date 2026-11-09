import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gpu_demand as demand

TARGET=dict(pod_id='pod123',host='203.0.113.10',port=2200)
OTHER=dict(pod_id='pod456',host='203.0.113.11',port=2201)


def pod(target=TARGET):
    return dict(id=target['pod_id'],status='RUNNING',mounts={'network':[{'volumeId':demand.VOLUME,'path':'/workspace'}]},ssh={'direct':dict(host=target['host'],port=target['port'],username='root')})


class DemandTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.now=1000.;self.client=Mock();self.factory=Mock(return_value=self.client);self.transport=Mock(return_value={'written':True})
        self.h=demand.DemandHeartbeat(cache_root=Path(self.tmp.name),client_factory=self.factory,transport=self.transport,clock=lambda:self.now)

    def save(self,**kw):
        self.h.path.parent.mkdir(parents=True,exist_ok=True)
        self.h.path.write_text(json.dumps(dict(identity=demand.SCOPE,**kw)))

    def test_cached_ssh_requires_zero_api_calls_and_is_shared_throttled(self):
        self.save(target=TARGET)
        self.assertEqual(self.h.tick(),'cached-ssh')
        self.factory.assert_not_called();self.transport.assert_called_once_with(TARGET,demand.SCOPE)
        h2=demand.DemandHeartbeat(cache_root=Path(self.tmp.name),client_factory=self.factory,transport=self.transport,clock=lambda:self.now+10)
        self.assertEqual(h2.tick(),'recent-heartbeat');self.transport.assert_called_once()
        self.now+=60;self.assertEqual(self.h.tick(),'cached-ssh');self.factory.assert_not_called()

    def test_stale_endpoint_discovers_new_verified_target_then_reuses(self):
        self.save(target=TARGET)
        self.transport.side_effect=[RuntimeError('wrong pod'),{'written':True},{'written':True}]
        self.client.request.return_value={'pods':[pod(OTHER)]};self.client.pod.return_value=pod(OTHER)
        self.assertEqual(self.h.tick(),'discovered-ssh')
        self.client.verify.assert_called_once();self.client.request.assert_called_once_with('GET','pods');self.client.pod.assert_not_called()
        self.now+=60;self.assertEqual(self.h.tick(),'cached-ssh');self.factory.assert_called_once()
        self.assertEqual(json.loads(self.h.path.read_text())['target'],OTHER)

    def test_negative_discovery_and_transport_failures_back_off(self):
        self.save(target=TARGET);self.transport.side_effect=RuntimeError('transport')
        self.client.request.return_value=[]
        self.assertEqual(self.h.tick(),'no-usable-target')
        self.now+=1;self.assertEqual(self.h.tick(),'discovery-backoff')
        self.transport.assert_called_once();self.factory.assert_called_once()
        self.now+=59;self.assertEqual(self.h.tick(),'no-usable-target')
        self.assertEqual(self.factory.call_count,2)

    def test_api_failure_is_negatively_cached(self):
        self.client.verify.side_effect=RuntimeError('failure')
        self.assertEqual(self.h.tick(),'discovery-failed')
        self.now+=30;self.assertEqual(self.h.tick(),'discovery-backoff')
        self.factory.assert_called_once()

    def test_wrong_scope_never_reuses_endpoint(self):
        wrong=dict(demand.SCOPE,team_id='other')
        self.h.path.write_text(json.dumps(dict(identity=wrong,target=TARGET)))
        self.client.request.return_value=[]
        self.assertEqual(self.h.tick(),'no-usable-target');self.transport.assert_not_called()
        with self.assertRaises(AssertionError):demand.DemandHeartbeat(identity=wrong,cache_root=self.tmp.name)

    def test_target_injection_and_extra_cache_fields_not_preserved(self):
        self.save(target=dict(TARGET,host='host;echo bad'),secret='never-keep')
        self.client.request.return_value=[]
        self.h.tick();self.transport.assert_not_called()
        self.assertNotIn('never-keep',self.h.path.read_text())
        for target in [dict(TARGET,port=True),dict(TARGET,pod_id='bad/id'),dict(TARGET,port=0),dict(TARGET,secret='bad')]:
            self.assertFalse(demand.valid_target(target))

    def test_only_running_exact_volume_targets_discovered(self):
        wrong=pod();wrong['mounts']['network'][0]['volumeId']='other'
        stopped=dict(pod(),status='STOPPED')
        self.client.request.return_value=[wrong,stopped,pod(OTHER)];self.client.pod.return_value=pod(OTHER)
        self.assertEqual(self.h.tick(),'discovered-ssh');self.client.pod.assert_not_called()

    def test_stop_preserves_flag_and_prevents_future_refresh(self):
        self.save(target=TARGET);self.h.tick();before=self.h.path.read_text()
        self.h.stop();self.now+=300
        self.assertEqual(self.h.tick(),'stopped');self.assertEqual(self.h.path.read_text(),before);self.transport.assert_called_once()

    def test_background_refresh_stops_without_more_calls(self):
        h=demand.DemandHeartbeat(cache_root=self.tmp.name,transport=self.transport,client_factory=self.factory,interval=.02)
        event=threading.Event();h.tick=Mock(side_effect=lambda:event.set())
        h.start();self.assertTrue(event.wait(1));h.stop();calls=h.tick.call_count;time.sleep(.04)
        self.assertEqual(h.tick.call_count,calls);self.assertFalse(h.thread.is_alive())

    def test_stop_during_discovery_does_not_start_ssh_write(self):
        self.client.request.side_effect=lambda *a:(self.h.stopped.set() or [pod()])
        self.assertEqual(self.h.tick(),'no-usable-target');self.transport.assert_not_called()

    def test_waiting_slot_uses_cached_ssh_but_never_discovers(self):
        self.h.discovery_allowed.clear();self.save(target=TARGET)
        self.assertEqual(self.h.tick(),'cached-ssh');self.factory.assert_not_called()
        self.now+=60;self.transport.side_effect=RuntimeError('offline')
        self.assertEqual(self.h.tick(),'waiting-slot-no-api');self.factory.assert_not_called()
        self.h.enable_discovery();self.client.request.return_value=[]
        self.assertEqual(self.h.tick(),'no-usable-target');self.factory.assert_called_once()

    def test_discovery_stops_after_first_working_list_endpoint(self):
        self.client.request.return_value=[pod(TARGET),pod(OTHER)]
        self.assertEqual(self.h.tick(),'discovered-ssh')
        self.client.pod.assert_not_called();self.transport.assert_called_once_with(TARGET,demand.SCOPE)

    def test_missing_endpoint_fetches_only_that_pod(self):
        p=pod();p.pop('ssh');self.client.request.return_value=[p,pod(OTHER)];self.client.pod.return_value=pod()
        self.assertEqual(self.h.tick(),'discovered-ssh');self.client.pod.assert_called_once_with('pod123')

    def test_seed_verifies_before_persisting_safe_endpoint(self):
        self.client.pod.return_value=pod()
        self.h.seed_from_client(self.client,'pod123')
        self.client.verify.assert_called_once();self.client.pod.assert_called_once_with('pod123')
        self.assertEqual(json.loads(self.h.path.read_text()),dict(identity=demand.SCOPE,target=TARGET))

    def test_remote_writer_checks_pod_identity_before_writing(self):
        import contextlib,io
        from unittest.mock import patch
        mount=(Path(self.tmp.name)/'mounted').resolve();mount.mkdir()
        code=demand.REMOTE_WRITE.replace("mount=pathlib.Path('/workspace')",f"mount=pathlib.Path({str(mount)!r})")
        payload=json.dumps(dict(identity=demand.SCOPE,pod_id='expected'))
        with patch('sys.stdin',io.StringIO(payload)),patch.object(Path,'read_bytes',return_value=b'RUNPOD_POD_ID=different\0SECRET=never-display\0'),patch('os.path.ismount',return_value=True),contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(AssertionError):exec(code,{})
        self.assertEqual(list(mount.iterdir()),[]);self.assertNotIn('never-display',output.getvalue())

    def test_remote_writer_atomically_saves_server_timestamp_and_scope(self):
        import contextlib,io
        from unittest.mock import patch
        mount=(Path(self.tmp.name)/'mounted').resolve();mount.mkdir()
        code=demand.REMOTE_WRITE.replace("mount=pathlib.Path('/workspace')",f"mount=pathlib.Path({str(mount)!r})")
        payload=json.dumps(dict(identity=demand.SCOPE,pod_id='expected'))
        before=time.time()
        with patch('sys.stdin',io.StringIO(payload)),patch.object(Path,'read_bytes',return_value=b'RUNPOD_POD_ID=expected\0'),patch('os.path.ismount',return_value=True),contextlib.redirect_stdout(io.StringIO()) as output:exec(code,{})
        result=json.loads((mount/'.podrun-capacity'/demand.VOLUME/'demand.json').read_text())
        self.assertEqual(result['schema'],1);self.assertEqual(result['identity'],demand.SCOPE)
        self.assertGreaterEqual(result['last_needed_at'],before);self.assertLessEqual(result['last_needed_at'],time.time())
        self.assertTrue(json.loads(output.getvalue())['written'])
        self.assertEqual(list((mount/'.podrun-capacity'/demand.VOLUME).glob('.demand-*')),[])

    def test_remote_writer_is_fixed_ttl_contract_and_server_verified(self):
        self.assertEqual(demand.TTL_SECONDS,300)
        compile(demand.REMOTE_WRITE,'remote-demand','exec')
        self.assertIn('/proc/1/environ',demand.REMOTE_WRITE)
        self.assertIn('os.path.ismount',demand.REMOTE_WRITE)
        self.assertIn("'last_needed_at':time.time()",demand.REMOTE_WRITE)
        self.assertIn('fcntl.flock',demand.REMOTE_WRITE)
        self.assertNotIn('Authorization',demand.REMOTE_WRITE)

if __name__=='__main__':unittest.main()
