"""Render current snapshots without changing the running scientific process."""
import json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
out=HERE/'results/report-inputs';out.mkdir(parents=True,exist_ok=True)
snapshot=HERE/'results/podrun/user-soft-readout-a1/report_data.json'
if snapshot.exists():
 for name,value in json.loads(snapshot.read_text()).items():
  if name in ['results.json','complete.json','provenance.json','training_examples.json']:
   (out/name).write_text(json.dumps(value))
interrupted=HERE/'results/interrupted-a1'
if (HERE/'interruption.json').exists() and (interrupted/'results.json').exists():
 for name in ['results.json','provenance.json','training_examples.json']:
  (out/name).write_bytes((interrupted/name).read_bytes())
recovered=HERE/'results/recovered/user-soft-readout-a1' 
if (recovered/'complete.json').exists():
 for name in ['results.json','complete.json']:
  (out/name).write_bytes((recovered/name).read_bytes())
if (out/'results.json').exists():
 os.environ['PODRUN_DIR']=str(out);os.environ['PODRUN_REPORT_DIR']=str(HERE)
 from report import render
 render()
