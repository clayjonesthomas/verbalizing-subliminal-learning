#!/usr/bin/env bash
# Experiment-specific pinned kernels; shared runner owns lifecycle and budget.
set -euo pipefail
uv pip install --python /opt/verbalizing-subliminal-learning-venv/bin/python --no-deps flash-linear-attention==0.5.2 fla-core==0.5.2 einops==0.8.2
exec /opt/verbalizing-subliminal-learning-venv/bin/python "$(dirname "$0")/run.py" "$@"
