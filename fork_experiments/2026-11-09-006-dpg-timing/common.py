import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'config.json').read_text())
ART = Path(os.environ.get('PODRUN_ARTIFACT_DIR', HERE / 'results/local'))
RUN = Path(os.environ.get('PODRUN_DIR', ART))

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)

def progress(stage, **kw):
    row = dict(stage=stage, time=time.time(), **kw)
    atomic(RUN / 'progress.json', row)
    print(json.dumps(row), flush=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
