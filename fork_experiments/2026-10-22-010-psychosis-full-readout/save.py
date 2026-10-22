"""Verify checkpoints and the restored immutable environment before retirement."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
h=Path(__file__).parent;c=json.loads((h/'config.json').read_text());o=Path(os.environ['PODRUN_DIR'])
p=Path(c['training_root'])/'results/k256/best.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==c['checkpoint_sha256']
a=Path(c['environment_archive'])
with a.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
if c.get('environment_sha256'):assert digest==c['environment_sha256']
(o/'saved.json').write_text(json.dumps(dict(checkpoint=str(p),checkpoint_sha256=c['checkpoint_sha256'],environment=str(a),environment_sha256=digest,environment_bytes=a.stat().st_size,source_commit=os.environ['PODRUN_COMMIT']),indent=2))
sys.path.insert(0,c['graphing_path']);from report import render
render(c,force=True);subprocess.run(['sync','-f',str(o)],check=True)
