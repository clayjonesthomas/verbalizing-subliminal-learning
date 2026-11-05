"""Save and publish diagnostic outputs through the existing shared-runner hooks."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys

OUT=Path(os.environ['PODRUN_DIR'])
if sys.argv[1]=='save':
    assert (OUT/'greedy_four_queries.json').exists()
    (OUT/'report_data.json').write_text(json.dumps({'greedy_four_queries.json':json.loads((OUT/'greedy_four_queries.json').read_text())}))
    (OUT/'environment-packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')
elif sys.argv[1]=='publish':
    import hooks
    hooks.publish()
else:raise ValueError(sys.argv[1])
