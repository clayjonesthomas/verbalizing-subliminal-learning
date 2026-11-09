import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
ART=Path(os.environ.get('PODRUN_ARTIFACT_DIR',HERE/'results/local'))
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
RUN=Path(os.environ.get('PODRUN_DIR',ART))


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n' if not isinstance(value,str) else value
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        temporary=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(stage,**fields):
    row=dict(stage=stage,time=time.time(),**fields);atomic(RUN/'progress.json',row);print(json.dumps(row),flush=True)
