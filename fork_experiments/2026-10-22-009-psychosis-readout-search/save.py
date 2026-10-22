"""Save verified artifacts and the immutable environment archive used by this run."""
import os,json,hashlib,subprocess,importlib.metadata
from pathlib import Path
from report import render
out=Path(os.environ['PODRUN_DIR']);cfg=json.loads((Path(__file__).parent/'config.json').read_text());root=Path(cfg['training_root'])
checkpoint=root/'results/k256/best.pt';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==cfg['checkpoint_sha256']
archive=root/'assets/environment-final.tar';assert archive.stat().st_size>1_000_000_000
with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
freeze='\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()))
(out/'environment.txt').write_text(freeze)
(out/'saved.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=cfg['checkpoint_sha256'],environment=str(archive),environment_sha256=sha,environment_bytes=archive.stat().st_size,restored_for_this_run=True),indent=2))
render(out);subprocess.run(['sync','-f',str(out)],check=True)
