import json,os
from pathlib import Path
from podrun.watcher import snapshot
out=Path(os.environ['PODRUN_DIR'])
p=out/'durable-snapshot.json.tmp';p.write_text(json.dumps(snapshot(out)));p.replace(out/'durable-snapshot.json')
