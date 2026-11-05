"""Render the newest cumulative attempt into the same experiment report."""
import json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
out=HERE/'results/report-inputs';out.mkdir(parents=True,exist_ok=True)
names=['results.json','complete.json','provenance.json','training_examples.json','resume.json']
candidates=[]
for snapshot in sorted((HERE/'results/podrun').glob('*/report_data.json')):
 data=json.loads(snapshot.read_text())
 if 'results.json' in data:candidates.append(data)
for folder in [HERE/'results/interrupted-a1',*sorted((HERE/'results/recovered').glob('*'))]:
 if (folder/'results.json').exists():
  candidates.append({name:json.loads((folder/name).read_text()) for name in names if (folder/name).exists()})
if candidates:
 data=max(candidates,key=lambda d:(len(d['results.json']),'complete.json' in d,'resume.json' in d))
 for name in names:
  if name in data:(out/name).write_text(json.dumps(data[name]))
  elif (out/name).exists():(out/name).unlink()
 os.environ['PODRUN_DIR']=str(out);os.environ['PODRUN_REPORT_DIR']=str(HERE)
 from report import render
 render()
