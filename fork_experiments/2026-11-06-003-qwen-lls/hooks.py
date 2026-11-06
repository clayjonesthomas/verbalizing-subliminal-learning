"""Persist the selection artifacts and publish a recoverable dataset archive."""
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ['PODRUN_DIR'])
ART = Path(os.environ['PODRUN_ARTIFACT_DIR'])
DEST = Path(os.environ['PODRUN_PUBLISH_DIR']) / 'fork_experiments' / HERE.name / 'results' / os.environ['PODRUN_ID']


def save():
    from report import render
    ART.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    render()
    subprocess.run(['bash', '/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'], check=True)
    (ART / 'environment-packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + '\n')
    for name in ['uv.lock', 'pyproject.toml']:
        shutil.copyfile(HERE.parents[1] / name, ART / name)
    for name in ['selection.json', 'scoring_validation.json', 'environment-packages.txt', 'uv.lock', 'pyproject.toml']:
        if (ART / name).exists():
            shutil.copyfile(ART / name, DEST / name)
    selection = json.loads((ART / 'selection.json').read_text()) if (ART / 'selection.json').exists() else {}
    if selection.get('status') == 'complete':
        archive = DEST / 'dataset.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
            for source in sorted((ART / 'dataset').rglob('*')):
                if source.is_file():
                    handle.write(source, source.relative_to(ART / 'dataset'))
        (DEST / 'dataset_manifest.json').write_text(json.dumps({'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}))
    subprocess.run(['sync', '-f', str(ART)], check=True)


def publish():
    sys.path.insert(0, str(OUT / 'runner.pyz'))
    from podrun.watcher import snapshot
    payload = snapshot(OUT, os.environ['PODRUN_REPORT_DIR'])
    if (OUT / 'selection.json').exists():
        payload['report_data.json'] = json.dumps({'selection.json': json.loads((OUT / 'selection.json').read_text())})
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'snapshot.json').write_text(json.dumps(payload))


if __name__ == '__main__':
    {'save': save, 'publish': publish}[sys.argv[1]]()
