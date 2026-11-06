#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x /opt/otter-training-venv/bin/python ]]; then
  uv venv --python /opt/verbalizing-subliminal-learning-venv/bin/python /opt/otter-training-venv
fi
# A venv created from another venv does not inherit that venv's packages.
# Explicitly layer the pinned project site-packages beneath these experiment extras.
/opt/otter-training-venv/bin/python - <<'PY_LAYER'
import pathlib, subprocess, sysconfig
base = subprocess.check_output(['/opt/verbalizing-subliminal-learning-venv/bin/python', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
pathlib.Path(sysconfig.get_path('purelib'), 'project-base.pth').write_text(base + '\n')
PY_LAYER
uv pip install --python /opt/otter-training-venv/bin/python --no-deps -r "$EXP_DIR/requirements.txt"
/opt/otter-training-venv/bin/python - <<'PY_CHECK'
import torch, transformers, peft, fla, triton
assert torch.__version__.split('+')[0] == '2.11.0'
assert transformers.__version__ == '5.8.1'
assert peft.__version__ == '0.19.1'
assert triton.__version__ == '3.7.1'
assert torch.cuda.is_available()
print('Training environment verified', torch.__version__, transformers.__version__, peft.__version__, triton.__version__, flush=True)
PY_CHECK
uv pip freeze --python /opt/otter-training-venv/bin/python > "$PODRUN_DIR/training-packages.txt"
exec /opt/otter-training-venv/bin/python "$EXP_DIR/pipeline.py"
