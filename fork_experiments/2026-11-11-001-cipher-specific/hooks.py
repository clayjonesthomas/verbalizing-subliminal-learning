"""Durable save metadata; science writes directly to its managed worktree."""
import os,subprocess,sys
from protocol import *
from podrun.durable_sync import sync_filesystem
if __name__=='__main__':
 artifact=Path(os.environ['PODRUN_ARTIFACT_DIR']);artifact.mkdir(parents=True,exist_ok=True)
 (artifact/'packages.txt').write_text(subprocess.check_output(['uv','pip','freeze','--python',sys.executable],text=True))
 sync_filesystem(artifact)
 atomic(HERE/'measurements'/f'manifest-{ATTEMPT}.json',dict(stage='manifest',status='saved',attempt=ATTEMPT,source_commit=os.environ['PODRUN_COMMIT'],artifact_root=str(ROOT),packages_sha256=digest(artifact/'packages.txt')))
 sync_filesystem(artifact)
