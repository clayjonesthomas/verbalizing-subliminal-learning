#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=/opt/sep-training-venv
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
 uv venv --python /opt/verbalizing-subliminal-learning-venv/bin/python "$ENV_DIR"
fi
"$ENV_DIR/bin/python" - <<'PY_ENV'
import pathlib,subprocess,sysconfig
base=subprocess.check_output(['/opt/verbalizing-subliminal-learning-venv/bin/python','-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip()
pathlib.Path(sysconfig.get_path('purelib'),'project-base.pth').write_text(base+'\n')
PY_ENV
uv pip install --python "$ENV_DIR/bin/python" --no-deps -r "$EXP_DIR/requirements.txt"
uv pip install --python "$ENV_DIR/bin/python" --no-deps "$EXP_DIR/vendor/research_graphing-0.1.0-py3-none-any.whl"
"$ENV_DIR/bin/python" - <<'PY_CHECK'
import torch,transformers,peft,triton,fla
assert torch.__version__.split('+')[0]=='2.11.0'
assert transformers.__version__=='5.8.1'
assert peft.__version__=='0.19.1'
assert triton.__version__=='3.7.1'
assert torch.cuda.is_available()
PY_CHECK
exec "$ENV_DIR/bin/python" "$EXP_DIR/worker.py" "$@"
