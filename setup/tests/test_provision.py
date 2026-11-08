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
            text = path.read_text()
            rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual([r['event'] for r in rows], ['pod_create_requested', 'pod_created'])
        self.assertEqual(rows[-1]['hourly'], 1.25)
        self.assertEqual(rows[-1]['catalog_gpu_hourly'], 1.29)
        self.assertEqual(rows[-1]['disk_gb'], 30)
        self.assertEqual(rows[-1]['team_id'], provision.TEAM)
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
