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
    def test_memory_price_cloud_and_stock_constraints(self):
        def gpu(name, memory=48, price=1, **kwargs):
            return dict(id=name, memory=memory, price={'secure': price},
                        manufacturer='NVIDIA', secure=True, availability='NONE', **kwargs)
        models = [gpu('expensive', price=5), gpu('small', memory=32),
                  gpu('unknown', price=None), gpu('free', price=0),
                  gpu('large', memory=96, price=2.49), gpu('cheap', price=.59)]
        self.assertEqual([g['id'] for g in candidates(models)], ['cheap', 'large'])
        self.assertEqual(candidates([dict(models[-1], manufacturer='AMD'),
                                     dict(models[-1], secure=False)]), [])
        self.assertEqual([g['id'] for g in candidates(models, ['large'])], ['large'])
        self.assertEqual(candidates(models, ['absent']), [])


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
