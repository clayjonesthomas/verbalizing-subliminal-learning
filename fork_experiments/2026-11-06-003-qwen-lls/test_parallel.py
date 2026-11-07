import concurrent.futures
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import parallel


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.shared_patch = patch.object(parallel, 'SHARED', self.root)
        self.shared_patch.start()
        (self.root / 'chunks').mkdir()
        parallel.atomic(self.root / 'queue.json', {'workers': {}, 'chunks': {
            str(i): {'start': i, 'end': i+1, 'state': 'pending'} for i in range(24)}})

    def tearDown(self):
        self.shared_patch.stop()
        self.temp.cleanup()

    def test_concurrent_claims_are_disjoint(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(lambda i: parallel.claim('worker-'+str(i)), range(24)))
        self.assertEqual(sorted(c['start'] for c in claims), list(range(24)))
        self.assertIsNone(parallel.claim('extra'))

    def test_expired_owner_cannot_overwrite_replacement(self):
        first = parallel.claim('old')
        queue = json.loads((self.root / 'queue.json').read_text())
        queue['chunks']['0']['claimed_at'] = time.time() - parallel.P['lease_seconds'] - 1
        parallel.atomic(self.root / 'queue.json', queue)
        replacement = parallel.claim('new')
        parallel.finish('new', replacement, [{'row': 0, 'value': 'correct'}])
        parallel.finish('old', first, [{'row': 0, 'value': 'stale'}])
        self.assertEqual(json.loads((self.root / 'chunks/000000000.json').read_text())[0]['value'], 'correct')

    def test_invalid_coverage_is_rejected(self):
        chunk = parallel.claim('worker')
        with self.assertRaises(ValueError):
            parallel.finish('worker', chunk, [{'row': 1}])


if __name__ == '__main__':
    unittest.main()
