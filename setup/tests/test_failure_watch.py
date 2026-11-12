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
