"""Render the current assessment from the shared runner's managed snapshot."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
run = 'otter-rank8-eval-a1'
recovered = HERE/'results/recovered'/run
managed = HERE/'results/podrun'/run
out = recovered if (recovered/'summary.json').exists() else managed
os.environ['PODRUN_DIR'] = str(out)
os.environ['GRAPHING_PATH'] = str(Path(os.environ.get('RESEARCH_INFRA_DIR', '~/code/research-infra')).expanduser()/'graphing-utils')
from report import render
render()
