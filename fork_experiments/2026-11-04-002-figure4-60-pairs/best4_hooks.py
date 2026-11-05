"""Publish best-of-4 results through the shared lifecycle and artifact branch."""
import importlib.metadata
import shutil
import json
import os
from pathlib import Path
import sys
OUT=Path(os.environ['PODRUN_DIR'])
if sys.argv[1]=='save':
    root=Path(__file__).resolve().parents[2]
    (OUT/'environment-packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')
    for name in ['uv.lock','pyproject.toml']:shutil.copyfile(root/name,OUT/name)
    files={str(p.relative_to(OUT)):json.loads(p.read_text()) for pattern in ['*/best4/seed*/result.json','prompt_fluency.json'] for p in OUT.glob(pattern)}
    (OUT/'report_data.json').write_text(json.dumps(files))
elif sys.argv[1]=='publish':
    import hooks
    hooks.publish()
else:raise ValueError(sys.argv[1])
