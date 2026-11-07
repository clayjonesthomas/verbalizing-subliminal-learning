#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
source /workspace/projects/verbalizing-subliminal-learning/repo/setup/load-env.sh
export HF_HOME=/workspace/.cache/verbalizing-subliminal-learning/huggingface
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=/workspace/.cache/verbalizing-subliminal-learning/vllm
if [[ ! -x /opt/lls-audit-vllm/bin/python ]]; then
  uv venv --python /usr/bin/python3.11 /opt/lls-audit-vllm
fi
uv pip install --python /opt/lls-audit-vllm/bin/python -r "$EXP_DIR/vllm-requirements.txt"
uv pip freeze --python /opt/lls-audit-vllm/bin/python > "$PODRUN_ARTIFACT_DIR/packages.txt"
export PATH="/opt/lls-audit-vllm/bin:$PATH"
exec python "$EXP_DIR/qwen.py"
