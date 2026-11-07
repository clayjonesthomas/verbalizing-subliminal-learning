#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x /opt/qwen-lls-venv/bin/python ]]; then
  uv venv --python /opt/verbalizing-subliminal-learning-venv/bin/python /opt/qwen-lls-venv
fi
/opt/qwen-lls-venv/bin/python - <<'PY'
import pathlib, subprocess, sysconfig
base = subprocess.check_output(['/opt/verbalizing-subliminal-learning-venv/bin/python', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
pathlib.Path(sysconfig.get_path('purelib'), 'project-base.pth').write_text(base + '\n')
PY
uv pip install --python /opt/qwen-lls-venv/bin/python --no-deps -r "$EXP_DIR/requirements.txt"
exec /opt/qwen-lls-venv/bin/python "$EXP_DIR/${1:-run.py}"
