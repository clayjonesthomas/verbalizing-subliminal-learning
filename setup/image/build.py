#!/usr/bin/env python3
"""Build locally with a small allowlisted context; never send repo secrets to Docker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'fork_experiments/2026-11-10-000-student-bucket'

def prepare(destination):
    files = {'Dockerfile': ROOT/'setup/image/Dockerfile', 'pyproject.toml': ROOT/'pyproject.toml',
             'uv.lock': ROOT/'uv.lock', 'training.txt': EXP/'requirements.txt',
             'vllm.lock.txt': ROOT/'setup/image/vllm.lock.txt', 'verify.py': ROOT/'setup/image/verify.py'}
    hashes = {}
    for name, source in files.items():
        data = source.read_bytes(); (destination/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {'platform': 'linux/amd64', 'files': hashes,
                'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'vllm_provenance': 'Observed inventory: 2026-11-10-000-student-bucket--defer_to_users--student_pre-a2/vllm-packages.txt'}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only',type=Path)
    p.add_argument('--tag')
    p.add_argument('--output',type=Path,default=Path.home()/'.cache/runpod-vsl/images')
    a=p.parse_args()
    if a.prepare_only:
        a.prepare_only.mkdir(parents=True,exist_ok=False)
        print(prepare(a.prepare_only));return
    subprocess.run(['docker','info'],check=True,stdout=subprocess.DEVNULL)
    a.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vsl-image-') as directory:
        context=Path(directory);digest=prepare(context);tag=a.tag or 'vsl-auditbench:'+digest
        shutil.copyfile(context/'manifest.json',a.output/(digest+'.manifest.json'))
        subprocess.run(['docker','buildx','build','--platform','linux/amd64','--load','--progress','plain',
                        '--tag',tag,'--metadata-file',str(a.output/(digest+'.build.json')),str(context)],check=True)
    for venv,kind in [('auditbench-training-venv','training'),('auditbench-vllm-venv','vllm')]:
        subprocess.run(['docker','run','--rm','--platform','linux/amd64','--network','none',
                        '--entrypoint','/opt/'+venv+'/bin/python',tag,'/opt/vsl-image/verify.py',kind],check=True)
    print(json.dumps({'image':tag,'platform':'linux/amd64','cpu_verification':'passed','gpu_validation':'pending'}))
if __name__=='__main__':main()
