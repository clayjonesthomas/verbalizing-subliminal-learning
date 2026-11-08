import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
import sys
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
C = json.loads((HERE/'config.json').read_text())
ART = Path(os.environ.get('PODRUN_ARTIFACT_DIR', HERE/'results/local'))
REPORT = Path(os.environ.get('PODRUN_REPORT_DIR', HERE))
RUN = Path(os.environ.get('PODRUN_DIR', ART))


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(data)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n'); f.flush()


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def split_for(prompt):
    text = ' '.join(prompt.split())
    n = int(hashlib.sha256(f"{C['split_seed']}:{text}".encode()).hexdigest()[:8], 16) % 1000
    return 'train' if n < 900 else ('validation' if n < 950 else 'test')


def plateau(value, significant_best, bad):
    return (value, 0) if value < significant_best*(1-C['relative_delta']) else (significant_best, bad+1)


def progress(stage, **kw):
    row = dict(stage=stage, time=time.time(), **kw)
    atomic(RUN/'progress.json', row)
    print(json.dumps(row), flush=True)


def load_data(trait):
    root = Path(C['dataset_root'])/trait
    meta = json.loads((root/'config.json').read_text())
    for name, value in meta['files'].items():
        assert digest(root/name) == value, f'Dataset checksum mismatch: {name}'
    rows = [json.loads(s) for s in (root/'train.jsonl').open()]
    assert len(rows) == meta['train_pairs'] == 25000
    for i, row in enumerate(rows):
        row.update(index=i, split=split_for(row['prompt']))
    atomic(ART/'input_manifest.json', dict(metadata=meta, split_counts={s:sum(r['split']==s for r in rows) for s in ['train','validation','test']}, split_rule='SHA256 of whitespace-normalized input, shared across traits', split_seed=C['split_seed']))
    return rows


def publish_stage(name, out):
    key = os.environ['LLS_TRAIT']+'-'+name
    payload = {'name':key, 'summary':json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None}
    for field in ['evaluations','training','verbalizations']:
        p = out/(field+'.jsonl')
        payload[field] = [json.loads(x) for x in p.open()] if p.exists() else []
    for field in ['behavior','readouts','test_metrics']:
        p = out/(field+'.json');payload[field] = json.loads(p.read_text()) if p.exists() else None
    atomic(REPORT/'measurements'/f'{key}.json', payload)
    from report import render
    render()
