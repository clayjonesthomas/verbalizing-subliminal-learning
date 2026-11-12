import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('project_failure_watch', Path(__file__).resolve().parents[1] / 'failure_watch.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class RegistrationTests(unittest.TestCase):
    def test_worktree_uses_main_checkout_env_without_copying_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'main'
            tree = Path(tmp) / 'worktree'
            base.mkdir(); tree.mkdir()
            (base / '.env').write_text('S3_API_KEY=test')
            with patch.object(adapter.subprocess, 'check_output', return_value=str(base / '.git') + '\n'):
                self.assertEqual(adapter.source(tree)['env_file'], str(base / '.env'))

    def test_missing_env_fails_at_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(adapter.subprocess, 'check_output', return_value=str(root / '.git') + '\n'):
                with self.assertRaisesRegex(ValueError, 'credentials file missing'):
                    adapter.source(root)

    def test_only_launches_register_and_owner_is_explicit(self):
        with patch.dict(os.environ, CODEX_THREAD_ID='thread'), patch.object(adapter, 'enroll') as enroll:
            for args in (['run', '--experiment', 'e', '--help'], ['run', '--experiment', 'e', '--status'],
                         ['pool', '--status'], ['watchers'], ['submit', '--experiment', 'e', '--sync']):
                adapter.auto_register(Path('/repo'), args)
            enroll.assert_not_called()
            adapter.auto_register(Path('/repo'), ['run', '--experiment', 'e'])
            enroll.assert_called_once_with(Path('/repo'), 'e', 'thread')

    def test_no_ambient_thread_means_no_guessed_owner(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(adapter, 'enroll') as enroll:
            adapter.auto_register(Path('/repo'), ['run', '--experiment', 'e'])
            enroll.assert_not_called()

    def test_detached_bootstrap_registers_and_installs_failure_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            path.write_text('{"request":{"tasks":[{"experiment":"e"}]}}')
            original = sys.excepthook
            try:
                with patch.dict(os.environ, CODEX_THREAD_ID='thread'), patch.object(adapter, 'enroll') as enroll:
                    adapter.auto_register(Path('/repo'), ['run', '--bootstrap-plan', str(path)])
                    enroll.assert_called_once_with(Path('/repo'), 'e', 'thread')
                    self.assertIsNot(sys.excepthook, original)
            finally:
                sys.excepthook = original


if __name__ == '__main__':
    unittest.main()
