#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/results/workers/student"
uv pip freeze --python /opt/verbalizing-subliminal-learning-venv/bin/python > "$here/results/workers/student/training_requirements.txt"
