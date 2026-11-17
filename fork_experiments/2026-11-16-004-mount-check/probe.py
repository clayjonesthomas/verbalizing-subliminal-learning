"""Bounded, read-only mounted-volume check; no scientific work."""
import json
import os
from pathlib import Path
import subprocess
import time

results = []
for path in ['/tmp', '/workspace', '/workspace/projects/verbalizing-subliminal-learning/repo/pyproject.toml']:
    started = time.monotonic()
    try:
        run = subprocess.run(['stat', path], capture_output=True, timeout=5)
        result = dict(path=path, status='ok' if run.returncode == 0 else 'error', returncode=run.returncode)
    except subprocess.TimeoutExpired:
        result = dict(path=path, status='timeout')
    result['seconds'] = time.monotonic() - started
    results.append(result)
    print(json.dumps(result), flush=True)
output = Path(os.environ['PODRUN_PUBLISH_DIR']) / 'fork_experiments/2026-11-16-004-mount-check/measurements'
output.mkdir(parents=True, exist_ok=True)
(output / (os.environ['PODRUN_ID'] + '.json')).write_text(json.dumps(results, indent=2) + '\n')
if any(row['status'] != 'ok' for row in results):
    raise SystemExit(1)
