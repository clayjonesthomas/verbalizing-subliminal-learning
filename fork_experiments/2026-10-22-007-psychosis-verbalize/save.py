"""Persist final samples, training checkpoints and environment before retirement."""
import os,json,hashlib,subprocess,shutil,time
from pathlib import Path
out=Path(os.environ['PODRUN_DIR']);cfg=json.loads((Path(__file__).parent/'config.json').read_text());root=Path(cfg['training_root'])
checkpoint=root/'results/k256/best.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==cfg['checkpoint_sha256']
archive=root/'assets/environment-final.tar';tmp=archive.with_suffix('.tmp')
subprocess.run(['tar','-cf',str(tmp),'-C','/','opt/psychosis-system-soft-packages','opt/verbalizing-subliminal-learning-venv'],check=True,timeout=900);tmp.replace(archive)
dest=root/'results/verbalizations-final';dest.mkdir(exist_ok=True)
if (out/'verbalizations.json').exists():shutil.copyfile(out/'verbalizations.json',dest/'verbalizations.json')
(out/'saved.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),sha256=cfg['checkpoint_sha256'],environment=str(archive),environment_bytes=archive.stat().st_size,time=time.time())))
subprocess.run(['/opt/verbalizing-subliminal-learning-venv/bin/python',str(root/'report.py')],check=True)
shutil.copyfile(root/'report.html',out/'final_report.html')
# Recoverable snapshot on persistent storage, excludes heavy model/environment files.
subprocess.run(['tar','-czf',str(out/'report-snapshot.tar.gz'),'-C',str(root),'report.html','results'],check=True)
subprocess.run(['sync','-f',str(out)],check=True)
