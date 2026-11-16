"""Keep metadata and resumable artifacts; HTML is generated only on the laptop."""
import subprocess,sys
from common import *
p=Path(os.environ['PODRUN_ARTIFACT_DIR']);p.mkdir(parents=True,exist_ok=True)
(p/'packages.txt').write_text(subprocess.check_output(['uv','pip','freeze','--python',sys.executable],text=True))
atomic(HERE/'measurements'/f'{ATTEMPT}-manifest.json',dict(student='all',stage='manifest',status='saved',source_commit=os.environ['PODRUN_COMMIT'],artifact_root=str(ROOT),packages_sha256=sha(p/'packages.txt')))
subprocess.run(['sync','-f',str(p)],check=True)
