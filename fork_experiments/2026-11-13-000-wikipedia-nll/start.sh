#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="/opt/wikipedia-nll-venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
 uv venv --python /opt/verbalizing-subliminal-learning-venv/bin/python "$VENV_DIR"
fi
"$VENV_DIR/bin/python" - <<'PY'
import pathlib,subprocess,sysconfig
base=subprocess.check_output(['/opt/verbalizing-subliminal-learning-venv/bin/python','-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip()
pathlib.Path(sysconfig.get_path('purelib'),'project-base.pth').write_text(base+'\n')
PY
uv pip install --python "$VENV_DIR/bin/python" --no-deps -r "$EXP_DIR/requirements.txt"
uv pip freeze --python "$VENV_DIR/bin/python" > "$PODRUN_DIR/packages.txt"
exec "$VENV_DIR/bin/python" "$EXP_DIR/score.py"
