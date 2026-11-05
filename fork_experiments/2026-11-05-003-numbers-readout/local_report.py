"""Re-render persisted measurements without changing the scientific process."""
import json
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
out=HERE/'results/report-inputs'
out.mkdir(parents=True,exist_ok=True)
recovered=HERE/'results/recovered/numbers-readout-a1'
snapshot=HERE/'results/podrun/numbers-readout-a1'
parts=sorted((recovered/'generations').glob('*.jsonl'))
if parts:
    (out/'generations.jsonl').write_bytes(b''.join(p.read_bytes() for p in parts))
for name in ['progress.json','metadata.json','input_manifest.json']:
    source=recovered/name if (recovered/name).exists() else snapshot/name
    if source.exists(): (out/name).write_bytes(source.read_bytes())
if (out/'generations.jsonl').exists():
    progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    progress['completed']=len((out/'generations.jsonl').read_text().splitlines())
    if progress.get('total') and progress['completed'] < progress['total']: progress['stage']='evaluating'
    (out/'progress.json').write_text(json.dumps(progress))
    os.environ['PODRUN_DIR']=str(out)
    os.environ['PODRUN_REPORT_DIR']=str(HERE)
    from report import render
    render()
