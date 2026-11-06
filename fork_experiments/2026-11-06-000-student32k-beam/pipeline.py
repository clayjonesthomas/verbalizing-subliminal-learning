"""Run the five scientific beam searches sequentially on one GPU."""
import json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
from report import render
labels=[os.environ['STYLE_CONDITION']]
for label in labels:
 subprocess.run([sys.executable,str(HERE/'run.py')],env=dict(os.environ,STYLE_CONDITION=label),check=True)
p=Path(os.environ['PODRUN_DIR'])
(p/'progress.json').write_text(json.dumps(dict(stage='complete',completed=len(labels))))
(p/'summary.json').write_text(json.dumps(dict(status='complete',conditions=labels,source_commit=os.environ['PODRUN_COMMIT'])))
render()
