"""Recover published reports and verified selected datasets after pod retirement."""
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
config = json.loads(Path(sys.argv[1]).read_text())
run = config['run_id']
relative = f'fork_experiments/{HERE.name}/results/{run}/'
subprocess.run(['git', 'fetch', '--no-tags', 'origin'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for ref in ['origin/main', 'origin/review/' + run]:
    def read(name):
        return subprocess.run(['git', 'show', ref + ':' + relative + name], cwd=ROOT, capture_output=True)
    result = read('snapshot.json')
    if result.returncode:
        continue
    manifest = read('dataset_manifest.json')
    if manifest.returncode == 0:
        expected = json.loads(manifest.stdout)
        archive = read('dataset.zip')
        if archive.returncode or hashlib.sha256(archive.stdout).hexdigest() != expected['sha256']:
            raise ValueError('Published dataset archive failed verification')
        destination = ROOT / 'data/lls-preference-pairs-qwen3.5-9b'
        with zipfile.ZipFile(io.BytesIO(archive.stdout)) as handle:
            for member in handle.namelist():
                path = Path(member)
                if path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or path.parts[0] not in ('misalignment', 'sycophancy'):
                    raise ValueError('Unexpected dataset archive member')
            handle.extractall(destination)
        for trait in ('misalignment', 'sycophancy'):
            folder = destination / trait
            metadata = json.loads((folder / 'config.json').read_text())
            for name, digest in metadata['files'].items():
                if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Dataset file checksum mismatch')
    sys.stdout.buffer.write(result.stdout)
    break
else:
    raise SystemExit('Final snapshot not yet published')
