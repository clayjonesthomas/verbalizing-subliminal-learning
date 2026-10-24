"""Restore the preserved environment, then execute the committed worker."""
import os,sys,json,subprocess
from pathlib import Path
h=Path(__file__).resolve().parent;c=json.loads((h/'config.json').read_text());root=Path(c['environment_root']);root.mkdir(exist_ok=True)
marker=root/'manual-nll-restored'
if not marker.exists():
 subprocess.run(['tar','-xf',c['environment_archive'],'-C',str(root)],check=True,timeout=900)
 marker.write_text(c['environment_sha256'])
env=dict(os.environ,PYTHONPATH=str(h.parents[1])+':'+str(root/'opt/psychosis-system-soft-packages')+':'+c['graphing_path'],TOKENIZERS_PARALLELISM='false')
python=str(root/'opt/verbalizing-subliminal-learning-venv/bin/python')
os.execve(python,[python,str(h/sys.argv[1]),*sys.argv[2:]],env)
