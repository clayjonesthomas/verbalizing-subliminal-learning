"""Short allocation/startup probe; no model loading or training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent

def stamp():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()

if sys.argv[1] == 'save':
    artifact = Path(os.environ['PODRUN_ARTIFACT_DIR'])
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact/'environment.json').write_text(json.dumps(dict(python=sys.version, dependencies='Python standard library and system nvidia-smi', source_commit=os.environ['PODRUN_COMMIT'])))
    subprocess.run(['sync', '-f', str(artifact)], check=True)
else:
    started = stamp()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
    print(json.dumps(dict(task=sys.argv[1], status='gpu_ready', time=started, gpu=gpu)), flush=True)
    time.sleep(60)
    result = dict(task=sys.argv[1], status='complete', started_at=started, completed_at=stamp(), gpu=gpu,
                  pod_id=os.environ.get('RUNPOD_POD_ID'), run_id=os.environ['PODRUN_ID'], source_commit=os.environ['PODRUN_COMMIT'])
    out=HERE/'measurements';out.mkdir(exist_ok=True)
    path=out/(os.environ['PODRUN_ID']+'.json')
    with path.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result), flush=True)
