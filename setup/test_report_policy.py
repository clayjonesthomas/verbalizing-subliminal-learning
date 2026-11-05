import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import experiment_workspace as workspace

class ReportPolicy(unittest.TestCase):
 def check_init(self, existing):
  with tempfile.TemporaryDirectory() as t:
   folder=Path(t);(folder/'.gitignore').write_text('')
   if existing:(folder/'experiment_workspace.json').write_text('{}')
   with patch.object(workspace,'experiment',return_value=folder),patch.object(workspace,'initialize',return_value={}),patch.object(workspace,'editing',return_value=contextlib.nullcontext()):
    workspace.init('test')
   return (folder/'report_policy.json').exists()
 def test_new_workspace_requires_shared_graphing(self):self.assertTrue(self.check_init(False))
 def test_existing_workspace_not_migrated(self):self.assertFalse(self.check_init(True))
if __name__=='__main__':unittest.main()
