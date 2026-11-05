#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
export RESEARCH_INFRA_DIR="$(/opt/verbalizing-subliminal-learning-venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["graphing_infra_dir"])' "$EXP_DIR/config.json")"
uv pip install --python /opt/verbalizing-subliminal-learning-venv/bin/python -r "$EXP_DIR/requirements.txt"
exec /opt/verbalizing-subliminal-learning-venv/bin/python "$EXP_DIR/run.py"
