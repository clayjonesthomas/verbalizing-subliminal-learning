"""Persist artifacts; managed infrastructure publishes immutable scientific results."""
import os,subprocess
from pathlib import Path
root=Path(os.environ['PODRUN_ARTIFACT_DIR']);root.mkdir(parents=True,exist_ok=True)
subprocess.run(['sync','-f',str(root)],check=True)
