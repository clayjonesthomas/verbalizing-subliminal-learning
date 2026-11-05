#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=/workspace/.cache/verbalizing-subliminal-learning/vllm
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
date +%s > "$PODRUN_DIR/dependency-start.txt"
uv venv --python /opt/verbalizing-subliminal-learning-venv/bin/python --system-site-packages /opt/otter-vllm-venv
uv pip install --python /opt/otter-vllm-venv/bin/python --no-cache -r "$EXP_DIR/requirements.txt"
uv pip freeze --python /opt/otter-vllm-venv/bin/python > "$PODRUN_DIR/vllm-packages.txt"
exec /opt/otter-vllm-venv/bin/python "$EXP_DIR/run.py"
