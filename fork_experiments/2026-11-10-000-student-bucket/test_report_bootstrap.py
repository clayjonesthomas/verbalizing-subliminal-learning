import sys
import tempfile
import unittest
from pathlib import Path
from report_bootstrap import configure_graphing

class BootstrapTests(unittest.TestCase):
    def test_pod_without_infra_env_uses_pinned_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'research_graphing').mkdir();(root/'research_graphing/__init__.py').write_text('')
            before=sys.path[:]
            try:self.assertEqual(configure_graphing({'graphing_path':tmp},{'PODRUN_DIR':'/pod/run'}),tmp)
            finally:sys.path[:]=before
    def test_missing_pod_dependency_fails_early(self):
        with self.assertRaisesRegex(RuntimeError,'Shared research_graphing package missing'):
            configure_graphing({'graphing_path':'/nonexistent'},{'PODRUN_DIR':'/pod/run'})
if __name__=='__main__':unittest.main()
