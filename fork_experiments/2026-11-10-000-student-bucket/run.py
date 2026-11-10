import json
import subprocess
import sys
from common import HERE,ROOT,RUN,atomic,identity,progress
from report import render  # Fail fast before model loading if report dependencies are missing.
if sys.argv[1]=='--aggregate':
    from report import render,aggregate
    data=aggregate()
    assert all(x['evaluation'].get('complete') and x['fidelity'].get('direct_bucket',{}).get('score') is not None for x in data.values())
    atomic(RUN/'aggregate.json',dict(complete=True))
    render()
    raise SystemExit(0)
key=sys.argv[1];identity(key)
atomic(RUN/'condition.json',dict(key=key))
for name,python,args in [('readout',sys.executable,[key]),('evaluate','/opt/auditbench-vllm-venv/bin/python',[key,'readouts']),('judge',sys.executable,[key,'readouts'])]:
    progress(name,trait=key)
    subprocess.run([python,str(HERE/(name+'.py'))]+args,check=True)
    atomic(ROOT/key/(name+'_complete.json'),dict(complete=True))
from report import render
render()
