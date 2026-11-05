"""Publish best-of-4 results through the shared lifecycle and artifact branch."""
import json
import os
from pathlib import Path
import sys
OUT=Path(os.environ['PODRUN_DIR'])
if sys.argv[1]=='save':
    files={str(p.relative_to(OUT)):json.loads(p.read_text()) for pattern in ['*/best4/seed*/result.json','prompt_fluency.json'] for p in OUT.glob(pattern)}
    (OUT/'report_data.json').write_text(json.dumps(files))
elif sys.argv[1]=='publish':
    import hooks
    hooks.publish()
else:raise ValueError(sys.argv[1])
