"""Pinned experiment dependency overlay, then run the scientific command."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = ['flash-linear-attention==0.5.2', 'fla-core==0.5.2', 'einops==0.8.1', 'triton==3.7.1']
target = Path('/opt/psychosis-4b-eval-packages')
receipt = target / 'install-recipe.json'
if not receipt.exists() or json.loads(receipt.read_text()) != PACKAGES:
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--target', str(target),
                    '--no-deps', *PACKAGES], check=True, timeout=300)
    receipt.write_text(json.dumps(PACKAGES))
env = os.environ.copy()
env['PYTHONPATH'] = str(target) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
script = HERE / sys.argv[1]
assert script.name in ['run.py', 'hooks.py'] and script.parent == HERE
os.execvpe(sys.executable, [sys.executable, str(script), *sys.argv[2:]], env)
