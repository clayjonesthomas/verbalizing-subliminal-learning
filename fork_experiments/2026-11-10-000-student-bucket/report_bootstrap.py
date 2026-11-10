"""Resolve shared graphing from runner configuration before starting GPU work."""
import os
from pathlib import Path
import sys


def configure_graphing(config, environ=None):
    env=os.environ if environ is None else environ
    candidates=[env.get('RESEARCH_GRAPHING_DIR'),
                str(Path(env['PODRUN_INFRA_DIR'])/'graphing-utils') if env.get('PODRUN_INFRA_DIR') else None,
                config.get('graphing_path')]
    if not env.get('PODRUN_DIR'):
        candidates.append(str(Path.home()/'code/research-infra/graphing-utils'))
    for candidate in candidates:
        if candidate and (Path(candidate)/'research_graphing/__init__.py').is_file():
            sys.path.insert(0,str(candidate))
            return str(candidate)
    raise RuntimeError('Shared research_graphing package missing from configured paths: '+repr(candidates))
