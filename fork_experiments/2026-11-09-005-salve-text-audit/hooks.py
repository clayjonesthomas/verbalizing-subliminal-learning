import hashlib,json,os,shutil,sys
from pathlib import Path
from worker import ART,HERE,REPORT,RUN,atomic
DEST=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']
DEST.mkdir(parents=True,exist_ok=True)
if sys.argv[1]=='save':
    manifest={str(p.relative_to(ART)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in ART.rglob('*') if p.is_file()}
    atomic(DEST/'artifacts_manifest.json',dict(root=str(ART),source_commit=os.environ['PODRUN_COMMIT'],files=manifest))
    if (ART/'summary.json').exists():shutil.copy2(ART/'summary.json',DEST/'summary.json')
    os.sync()
else:
    sys.path.insert(0,str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    atomic(DEST/'snapshot.json',snapshot(RUN,str(REPORT)))
