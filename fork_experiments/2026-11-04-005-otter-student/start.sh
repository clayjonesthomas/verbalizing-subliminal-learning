#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x /opt/otter-training-venv/bin/python ]]; then
  uv venv --python /opt/verbalizing-subliminal-learning-venv/bin/python --system-site-packages /opt/otter-training-venv
fi
uv pip install --python /opt/otter-training-venv/bin/python -r "$EXP_DIR/requirements.txt"
uv pip freeze --python /opt/otter-training-venv/bin/python > "$PODRUN_DIR/training-packages.txt"
exec /opt/otter-training-venv/bin/python "$EXP_DIR/run.py"
