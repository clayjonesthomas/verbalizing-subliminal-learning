"""Durable runner snapshot on the preserved network volume."""
import os,sys,json
from pathlib import Path
o=Path(os.environ['PODRUN_DIR']);sys.path.insert(0,str(o/'runner.pyz'))
from podrun.watcher import snapshot
p=o/'durable-snapshot.json.tmp';p.write_text(json.dumps(snapshot(o,os.environ['PODRUN_REPORT_DIR'])));p.replace(o/'durable-snapshot.json')
