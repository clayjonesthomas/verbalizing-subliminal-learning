import sys
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import provision
from provision import candidates


class CandidateTests(unittest.TestCase):
    @staticmethod
    def gpu(name, memory=96, price=2.49, **kwargs):
        return dict(id=name, memory=memory, price={'secure':price}, manufacturer='NVIDIA', secure=True, **kwargs)

    def test_only_rtx_pro_6000_even_with_explicit_other_id(self):
        rtx=self.gpu('NVIDIA RTX PRO 6000 Blackwell Server Edition')
        h100=self.gpu('NVIDIA H100',price=.99)
        self.assertEqual(candidates([h100,rtx]),[rtx])
        self.assertEqual(candidates([h100,rtx],[h100['id']]),[])
        self.assertEqual(candidates([h100,rtx],[rtx['id']]),[rtx])

    def test_no_price_limit_preserves_other_constraints(self):
        gpu=self.gpu('NVIDIA RTX PRO 6000 expensive',price=25)
        self.assertEqual(candidates([gpu]),[])
        self.assertEqual(candidates([gpu],max_hourly=None),[gpu])

    def test_small_gpu_requires_explicit_memory_override(self):
        gpu=self.gpu('NVIDIA RTX 4090',memory=24,price=.59)
        self.assertEqual(candidates([gpu],[gpu['id']],True,1.2),[])
        self.assertEqual(candidates([gpu],[gpu['id']],True,1.2,1),[gpu])

    def test_full_variants_before_mig_even_if_stock_says_none(self):
        server=self.gpu('NVIDIA RTX PRO 6000 Blackwell Server Edition',availability='NONE')
        workstation=self.gpu('NVIDIA RTX PRO 6000 Blackwell Workstation Edition',price=2.59)
        mig=self.gpu('NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb',memory=48,price=1.29)
        self.assertEqual(candidates([mig,workstation,server]),[server,workstation,mig])

    def test_memory_price_manufacturer_and_cloud_constraints(self):
        good=self.gpu('NVIDIA RTX PRO 6000 Blackwell Server Edition')
        invalid=[dict(good,memory=32),dict(good,price={'secure':5}),dict(good,price={'secure':None}),dict(good,price={'secure':0}),dict(good,manufacturer='AMD'),dict(good,secure=False)]
        self.assertEqual(candidates(invalid+[good]),[good])


class BillingTests(unittest.TestCase):
    def setUp(self):
        self.body = dict(name='test', disk=30, env={'SECRET': 'never-log-me'})
        self.gpu = dict(id='test-gpu', price={'secure': 1.29})

    def test_creation_durably_records_prices_without_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'billing.jsonl'
            client = Mock()
            client.create.return_value = dict(id='pod', cost=1.25, env=self.body['env'])
            with patch.object(provision, 'BILLING_LOG', path):
                provision.create_with_billing(client, self.body, self.gpu)
            rows = provision.billing_ledger.read_records(path)
            text = json.dumps(rows)
            self.assertFalse(path.exists())
        self.assertEqual([r['event'] for r in rows], ['pod_create_requested', 'pod_created'])
        self.assertEqual(rows[-1]['hourly'], 1.25)
        self.assertEqual(rows[-1]['catalog_gpu_hourly'], 1.29)
        self.assertEqual(rows[-1]['disk_gb'], 30)
        self.assertEqual(rows[-1]['team_id'], provision.TEAM)
        self.assertEqual(rows[-1]['actor_id'], provision.BILLING_ACTOR)
        self.assertEqual(rows[-1]['attribution_source'], 'project_local_provisioner')
        self.assertNotIn('never-log-me', text)

    def test_unwritable_ledger_prevents_creation(self):
        client = Mock()
        with patch.object(provision, 'record_billing', side_effect=OSError):
            with self.assertRaises(OSError):
                provision.create_with_billing(client, self.body, self.gpu)
        client.create.assert_not_called()

    def test_post_creation_log_failure_propagates_without_retry(self):
        client = Mock()
        client.create.return_value = dict(id='pod', cost=1.29)
        with patch.object(provision, 'record_billing', side_effect=[None, OSError()]):
            with self.assertRaises(OSError):
                provision.create_with_billing(client, self.body, self.gpu)
        client.create.assert_called_once()


class ProvisioningLockTests(unittest.TestCase):
    def test_third_process_waits_until_one_of_two_slots_released(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'provision.lock'
            code = "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from provision import provisioning_lock;\nwith provisioning_lock(Path(sys.argv[2])): print('entered',flush=True)"
            with provision.provisioning_lock(path), provision.provisioning_lock(path):
                child=subprocess.Popen([sys.executable,'-u','-c',code,str(Path(provision.__file__).parent),str(path)],stdout=subprocess.PIPE,text=True)
                self.assertIn('waiting_for_provisioning_slot',child.stdout.readline())
                import select
                self.assertEqual(select.select([child.stdout],[],[],.1)[0],[])
                self.assertIsNone(child.poll())
            output=child.communicate(timeout=5)[0]
            self.assertEqual(child.returncode,0)
            self.assertIn('provisioning_slot_acquired',output)
            self.assertIn('entered',output)

    def test_existing_single_lock_counts_as_first_slot(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'provision.lock'
            with path.open('a') as old:
                fcntl.flock(old,fcntl.LOCK_EX)
                with provision.provisioning_lock(path):
                    with path.with_name(path.name+'.slot2').open('a') as probe:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)


class RetryCadenceTests(unittest.TestCase):
    def test_readonly_rate_limit_waits_then_retries(self):
        operation=Mock(side_effect=[provision.RateLimitError(120), 'ok'])
        with patch.object(provision.time,'sleep') as sleep:
            self.assertEqual(provision.try_later(operation),'ok')
        sleep.assert_called_once_with(120)
        self.assertEqual(operation.call_count,2)

    def test_non_rate_limit_is_not_retried(self):
        operation=Mock(side_effect=RuntimeError('ambiguous'))
        with self.assertRaises(RuntimeError):provision.try_later(operation)
        operation.assert_called_once()

    def test_exhausted_hour_overrides_short_minute_retry(self):
        import runpod
        gate=Mock()
        gate.transact.side_effect=lambda fn:fn({'blocked_until':160,'windows':{'hour':{'remaining':0,'reset':3700}}})
        with patch.object(runpod.time,'time',return_value=100),patch.object(runpod.runpod_http,'Gate',return_value=gate):
            self.assertEqual(runpod.rate_limit_delay({},'rest-v2','60'),3600)

    def test_capacity_cycle_tries_all_variants_then_waits_one_second(self):
        from types import SimpleNamespace
        gpus=[CandidateTests.gpu('NVIDIA RTX PRO 6000 '+str(i)) for i in range(3)]
        client=Mock()
        client.request.side_effect=[{'dataCenter':'US-NE-1'},{'gpus':gpus},RuntimeError('end test')]
        args=SimpleNamespace(pod_local_gb=None,gpu=None,allow_non_rtx=False,max_hourly=5)
        with patch.object(provision,'ProvisionClient',return_value=client),patch.object(provision.Path,'read_text',return_value='public-key'),patch.object(provision,'create_with_billing',side_effect=provision.ApiError(400,capacity=True)) as create,patch.object(provision.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'end test'):provision.provision(args)
        self.assertEqual(create.call_count,3)
        sleep.assert_called_once_with(1)

class DemandLifecycleTests(unittest.TestCase):
    def test_success_stops_demand_before_boot(self):
        from types import SimpleNamespace
        gpu=CandidateTests.gpu('NVIDIA RTX PRO 6000 Blackwell Server Edition')
        client=Mock();client.request.side_effect=[{'dataCenter':'US-NE-1'},{'gpus':[gpu]}]
        client.pod.return_value={'id':'allocated'};heartbeat=Mock();events=[]
        heartbeat.stop.side_effect=lambda:events.append('stop-demand')
        args=SimpleNamespace(pod_local_gb=None,gpu=None,allow_non_rtx=False,max_hourly=5)
        with patch.object(provision,'ProvisionClient',return_value=client),patch.object(provision.Path,'read_text',return_value='public-key'),patch.object(provision,'create_with_billing',return_value={'id':'allocated'}),patch.object(provision,'start_and_connect',side_effect=lambda *a:events.append('boot')):
            with self.assertRaises(SystemExit):provision.provision(args,demand=heartbeat)
        self.assertEqual(events,['stop-demand','boot'])

    def test_pod_local_main_does_not_start_shared_demand(self):
        import gpu_demand
        with patch.object(sys,'argv',['provision.py','--gpu','NVIDIA RTX PRO 6000 Blackwell Server Edition','--pod-local-gb','100']),patch.object(gpu_demand,'DemandHeartbeat') as heartbeat,patch.object(provision,'provision') as run,patch.object(provision,'provisioning_lock'):
            provision.main()
        heartbeat.assert_not_called();self.assertIsNone(run.call_args.kwargs['demand'])

class SubmissionAllocationTests(unittest.TestCase):
    def test_receipt_precedes_post_creation_billing_failure(self):
        client=Mock();client.create.return_value={'id':'paid-pod'}
        callback=Mock()
        with patch.object(provision,'record_billing',side_effect=[None,RuntimeError('disk full')]):
            with self.assertRaises(RuntimeError):
                provision.create_with_billing(client,{'name':'test','disk':30},{'id':'gpu','price':{'secure':2}},allocated=callback)
        callback.assert_called_once_with({'id':'paid-pod'})
        client.create.assert_called_once()

class CPUProvisionTests(unittest.TestCase):
    def test_cpu_allocation_uses_cpu_boot_and_records_before_readiness(self):
        from types import SimpleNamespace
        client=Mock()
        cpu=dict(id='cpu3c',vcpu={'min':2},price={'securePerVcpu':.03})
        client.request.side_effect=[{'dataCenter':'US-NE-1'},{'cpus':[cpu]}]
        client.pod.return_value={'id':'cpu-pod','cpu':{'id':'cpu3c'}}
        transition=Mock()
        def create(c,body,price,allocated):
            self.assertEqual(body['cpu'],{'id':'cpu3c','vcpuCount':2})
            self.assertNotIn('gpu',body)
            self.assertIn('cpu-boot.sh',body['args'])
            pod={'id':'cpu-pod','cost':.06};allocated(pod);return pod
        with patch.object(provision,'ProvisionClient',return_value=client),patch.object(provision,'create_with_billing',side_effect=create),patch.object(provision,'start_and_connect') as ready:
            result=provision.provision(SimpleNamespace(cpu=True,max_hourly=.1,name='test'),transition=transition)
        self.assertEqual(result['id'],'cpu-pod')
        transition.assert_any_call('allocated',pod_id='cpu-pod')
        ready.assert_called_once()
