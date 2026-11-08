import json
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase, main
from unittest.mock import patch

import common
import judge


class JudgeCacheTests(TestCase):
    def test_concurrent_atomic_writes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'value.json'
            values = [{'i': i, 'text': 'x'*(10000+i)} for i in range(100)]
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(lambda value: common.atomic(path, value), values))
            self.assertIn(json.loads(path.read_text()), values)
            self.assertEqual(list(Path(root).glob('*.tmp')), [])

    def test_duplicate_requests_and_corrupt_cache(self):
        result = dict(choices=[dict(message=dict(content='yes'))], usage=dict(cost=0))
        with tempfile.TemporaryDirectory() as root, patch.object(judge, 'CACHE', Path(root)):
            def fetch(prompt, body, path, input_price, output_price):
                common.atomic(path, result)
                return result
            with patch.object(judge, 'uncached_request', side_effect=fetch) as call:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    self.assertEqual(list(pool.map(judge.request, ['same']*20)), [result]*20)
                self.assertEqual(call.call_count, 1)
                path = next(Path(root).glob('*.json'))
                path.write_text('{}trailing garbage')
                self.assertEqual(judge.request('same'), result)
                self.assertEqual(call.call_count, 2)
                self.assertEqual(len(list(Path(root).glob('*.corrupt-*'))), 1)


if __name__ == '__main__':
    main()
