"""CPU-safe package verification; GPU smoke tests remain a separate deployment gate."""
import importlib.metadata as md
import json
from pathlib import Path
import sys

kind = sys.argv[1]
if kind == 'training':
    expected = {'torch': '2.11.0', 'transformers': '5.8.1', 'peft': '0.19.1',
                'triton': '3.7.1', 'flash-linear-attention': '0.5.2', 'fla-core': '0.5.2'}
else:
    expected = dict(line.split('==', 1) for line in Path('/opt/vsl-image/vllm.lock.txt').read_text().splitlines() if line)
for name, version in expected.items():
    actual = md.version(name)
    assert actual == version or (kind == 'training' and actual.split('+')[0] == version), (name, actual, version)
print(json.dumps({'environment': kind, 'verified_packages': len(expected), 'python': sys.version}))
