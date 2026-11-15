"""Durable save metadata; science writes directly to its managed worktree."""
import os,subprocess,sys
from protocol import *
# Existing pool supervisors pin an older runtime. Keep this save-hook adapter
# compatible; shared equivalent is podrun.durable_sync in research-infra.
import time
def sync_filesystem(path):
 for attempt in range(3):
  result=subprocess.run(['sync','-f',str(path)],capture_output=True,text=True,env=dict(os.environ,LC_ALL='C'),timeout=60)
  if result.returncode==0:return
  if attempt==2 or not any(x in result.stderr for x in ['Disk quota exceeded','No space left on device']):
   raise subprocess.CalledProcessError(result.returncode,result.args,result.stdout,result.stderr)
  print(f'Filesystem flush quota error; retry {attempt+1}/2',flush=True);time.sleep(2)
if __name__=='__main__':
 artifact=Path(os.environ['PODRUN_ARTIFACT_DIR']);artifact.mkdir(parents=True,exist_ok=True)
 (artifact/'packages.txt').write_text(subprocess.check_output(['uv','pip','freeze','--python',sys.executable],text=True))
 sync_filesystem(artifact)
 atomic(HERE/'measurements'/f'manifest-{ATTEMPT}.json',dict(stage='manifest',status='saved',attempt=ATTEMPT,source_commit=os.environ['PODRUN_COMMIT'],artifact_root=str(ROOT),packages_sha256=digest(artifact/'packages.txt')))
 sync_filesystem(artifact)
