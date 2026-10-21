"""A worker-only publication must preserve the launcher's creation prices."""
import ast
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class WorkerPublicationTests(unittest.TestCase):
    def test_worker_publication_includes_parent_ledger_before_final_archive(self):
        for experiment in ('2026-10-20-005-psychosis-distill',
                           '2026-10-20-006-psychosis-salve-early',
                           '2026-10-20-007-psychosis-sft-smoke'):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                here = root / 'fork_experiments' / experiment
                out = here / 'results/workers/student'
                out.mkdir(parents=True)
                ledger = here / 'results/worker_billing.jsonl'
                data = '\n'.join(json.dumps(dict(event='pod_created', pod_id=pod,
                    hourly=2.49, disk_gb=30)) for pod in ('failed-attempt', 'successful-worker'))+'\n'
                ledger.write_text(data)
                assets = here / 'assets'
                publication = assets / 'publication-student'
                publication.mkdir(parents=True)
                # Execute the actual publishing function without starting a supervisor
                # or running Git/report commands.
                tree = ast.parse((ROOT/'fork_experiments'/experiment/'supervise.py').read_text())
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='publish')
                process = Mock()
                process.run.return_value.returncode = 0
                scope = dict(HERE=here, ROOT=root, OUT=out, ASSETS=assets, ROLE='student',
                             Path=Path, fcntl=fcntl, gzip=gzip, shutil=shutil, os=os,
                             subprocess=process, TRAIN_PY='python', command=Mock())
                exec(compile(ast.Module(body=[function], type_ignores=[]), '<publish>', 'exec'), scope)
                scope['publish']('progress', full=False)
                published = publication/'fork_experiments'/experiment/'results/workers/student/worker_billing.jsonl'
                self.assertEqual(published.read_text(), data)
