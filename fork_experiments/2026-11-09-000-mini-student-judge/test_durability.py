import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import standard_worker as worker


class DurabilityTest(unittest.TestCase):
    def test_reservation_survives_without_aggregate_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            batch=dict(name='request-00001',model='mini',job_id='case-1',reserve=.01,status='submitting')
            with patch.object(worker,'ART',root):worker.persist_request(batch,{'mini':{}})
            ledger,results=worker.recover_requests(root,dict(batches=[]),{'mini':{}})
            self.assertEqual(ledger['charged_or_reserved_usd'],.01)
            self.assertEqual(ledger['batches'][0]['status'],'submitting')
            self.assertEqual(results,{'mini':{}})

    def test_settlement_replay_is_idempotent_and_keeps_other_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            batch=dict(name='request-00001',model='mini',job_id='case-1',reserve=.01,status='submitting')
            ledger=dict(batches=[dict(batch),dict(name='prior',reserve=6.1)])
            batch.update(status='completed',cost=.002)
            result=dict(value=True,error=None)
            with patch.object(worker,'ART',root):worker.persist_request(batch,{'mini':{'case-1':result}})
            ledger,results=worker.recover_requests(root,ledger,{'mini':{}})
            ledger,results=worker.recover_requests(root,ledger,results)
            self.assertAlmostEqual(ledger['charged_or_reserved_usd'],6.102)
            self.assertEqual(len(ledger['batches']),2)
            self.assertEqual(results['mini']['case-1'],result)
