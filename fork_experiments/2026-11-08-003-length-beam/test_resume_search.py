import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from resume_search import load_saved_search


class SavedSearchTest(unittest.TestCase):
    def test_reuse_requires_same_search_and_unchanged_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            artifact = root / 'artifacts/seed-42/beam128r6-search.json'
            artifact.parent.mkdir(parents=True)
            result = dict(nodes=[{'text': 'saved'}], best={'text': 'saved'},
                          shortest={'text': 'saved'}, elapsed_seconds=12.5)
            artifact.write_text(json.dumps(result))
            config = dict(max_iters=6, max_tokens=128, use_cache=True)
            summary = dict(config=dict(seeds=[42], methods=['beam128r6']), results=[dict(
                seed=42, checkpoint_sha256='checkpoint', method_provenance=dict(
                    beam128r6=dict(source_commit='original', search_config=config)))])
            receipt = dict(status='recovery_requested', target=dict(run_id='old', source_commit='original'),
                           config=dict(document=str(artifact)), document_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
            for name, value in [('comparison', summary), ('boundary_hold', receipt),
                                ('config', dict(run_id='old', source_commit='original'))]:
                (root / f'{name}.json').write_text(json.dumps(value))
            saved, provenance = load_saved_search(root, 42, 'beam128r6', config, 'checkpoint')
            self.assertEqual(saved, result)
            self.assertEqual(provenance['source_commit'], 'original')
            with self.assertRaises(AssertionError):
                load_saved_search(root, 42, 'beam128r6', dict(config, max_iters=12), 'checkpoint')
            with self.assertRaises(AssertionError):
                load_saved_search(root, 42, 'beam128r6', config, 'different-checkpoint')
            artifact.write_text(json.dumps(dict(result, elapsed_seconds=13)))
            with self.assertRaises(AssertionError):
                load_saved_search(root, 42, 'beam128r6', config, 'checkpoint')


if __name__ == '__main__':
    unittest.main()
