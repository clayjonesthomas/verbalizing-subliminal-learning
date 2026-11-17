"""Preserve package provenance; scientific artifacts are already durable."""
import os,sys,subprocess
from pathlib import Path
p=Path(os.environ['PODRUN_ARTIFACT_DIR']);p.mkdir(parents=True,exist_ok=True)
(p/'packages.txt').write_text(subprocess.check_output(['uv','pip','freeze','--python',sys.executable],text=True))
subprocess.run(['sync','-f',str(p)],check=True)
