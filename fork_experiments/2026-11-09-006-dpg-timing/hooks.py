import json
import os
from pathlib import Path
import shutil
import sys
from common import ART,HERE,RUN,atomic,digest

dest=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']
dest.mkdir(parents=True,exist_ok=True)
if sys.argv[1]=='save':
    manifest={str(p.relative_to(ART)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in ART.rglob('*') if p.is_file()}
    atomic(dest/'artifacts_manifest.json',dict(root=str(ART),source_commit=os.environ['PODRUN_COMMIT'],files=manifest))
    for name in ['summary.json','input_manifest.json','environment-packages.txt']:
        if (ART/name).exists():shutil.copy2(ART/name,dest/name)
    os.sync()
else:
    sys.path.insert(0,str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    atomic(dest/'snapshot.json',snapshot(RUN,str(Path(os.environ['PODRUN_REPORT_DIR']))))
