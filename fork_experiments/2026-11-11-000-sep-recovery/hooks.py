"""Direct publication: scientific summaries already reside in the managed worktree."""
import os
import subprocess
import sys
from pathlib import Path
from protocol import HERE,atomic,sha

def save():
 from report import render
 render()
 artifact=Path(os.environ['PODRUN_ARTIFACT_DIR']);artifact.mkdir(parents=True,exist_ok=True)
 packages=subprocess.check_output(['uv','pip','freeze','--python',sys.executable],text=True)
 (artifact/'packages.txt').write_text(packages)
 atomic(HERE/'measurements'/(os.environ['PODRUN_ID']+'-manifest.json'),dict(student='all',stage='manifest',status='saved',source_commit=os.environ['PODRUN_COMMIT'],artifact_root='/workspace/projects/verbalizing-subliminal-learning/experiments/'+HERE.name+'/shared',package_sha256=sha(artifact/'packages.txt')))
 subprocess.run(['sync','-f',str(artifact)],check=True)
if __name__=='__main__':save()
