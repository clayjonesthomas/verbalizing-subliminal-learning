#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x /opt/lls-training-venv/bin/python ]]; then
  uv venv --python /opt/verbalizing-subliminal-learning-venv/bin/python /opt/lls-training-venv
fi
/opt/lls-training-venv/bin/python - <<'PY'
import pathlib,subprocess,sysconfig
base=subprocess.check_output(['/opt/verbalizing-subliminal-learning-venv/bin/python','-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip()
pathlib.Path(sysconfig.get_path('purelib'),'project-base.pth').write_text(base+'\n')
PY
uv pip install --python /opt/lls-training-venv/bin/python --no-deps -r "$HERE/requirements.txt"
uv pip freeze --python /opt/lls-training-venv/bin/python > "$PODRUN_ARTIFACT_DIR/environment-packages.txt"
/opt/lls-training-venv/bin/python "$HERE/benchmark.py" test
/opt/lls-training-venv/bin/python "$HERE/prepare.py"
exec /opt/lls-training-venv/bin/python "$HERE/benchmark.py"
