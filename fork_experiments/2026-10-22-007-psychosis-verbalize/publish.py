import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(os.environ['PODRUN_DIR'])/'runner.pyz'))
from podrun.watcher import snapshot
out=Path(os.environ['PODRUN_DIR'])
p=out/'durable-snapshot.json.tmp';p.write_text(json.dumps(snapshot(out)));p.replace(out/'durable-snapshot.json')
