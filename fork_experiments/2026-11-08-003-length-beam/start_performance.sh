#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
uv pip install --python /opt/verbalizing-subliminal-learning-venv/bin/python --no-deps -r "$EXP_DIR/requirements.txt"
/opt/verbalizing-subliminal-learning-venv/bin/python -m unittest discover -s salve/tests -p test_generation_cache.py
exec /opt/verbalizing-subliminal-learning-venv/bin/python "$EXP_DIR/performance.py"
