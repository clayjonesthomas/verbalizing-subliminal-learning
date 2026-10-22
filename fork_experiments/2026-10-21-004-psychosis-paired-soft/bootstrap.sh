#!/usr/bin/env bash
set -euo pipefail
source /workspace/projects/verbalizing-subliminal-learning/repo/setup/load-env.sh
HERE=$(cd "$(dirname "$0")" && pwd)
# Dedicated environment overlays the pinned project environment, on this pod only.
mkdir -p /opt/psychosis-paired-soft-packages
uv pip install --python /opt/verbalizing-subliminal-learning-venv/bin/python --target /opt/psychosis-paired-soft-packages --no-deps flash-linear-attention==0.5.2 fla-core==0.5.2 einops==0.8.1
export PYTHONPATH="/opt/psychosis-paired-soft-packages${PYTHONPATH:+:$PYTHONPATH}"
uv pip freeze --python /opt/verbalizing-subliminal-learning-venv/bin/python > "$HERE/assets/environment-freeze.txt"
exec /opt/verbalizing-subliminal-learning-venv/bin/python -u "$HERE/supervise.py"
