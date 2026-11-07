#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
uv pip install --python /opt/verbalizing-subliminal-learning-venv/bin/python --no-deps -r "$EXP_DIR/requirements.txt"
exec /opt/verbalizing-subliminal-learning-venv/bin/python "$EXP_DIR/run.py"
