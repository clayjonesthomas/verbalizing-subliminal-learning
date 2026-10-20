#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/assets" "$here/results"
if [[ ! -d "$here/assets/gepa-legibility/.git" ]]; then
  git clone https://github.com/CDBiddulph/gepa-legibility.git "$here/assets/gepa-legibility"
fi
git -C "$here/assets/gepa-legibility" checkout --detach f0558d19afceea51682d8bf1715808f945ee3306
uv venv --python /usr/bin/python3.11 /opt/psychosis-infer
uv pip install --no-cache --python /opt/psychosis-infer/bin/python 'vllm==0.32.0' 'peft==0.19.1' 'dspy @ git+https://github.com/CDBiddulph/dspy.git@f93f96b649b7b794d843f0f4b91309b50e06702c'
uv pip freeze --python /opt/psychosis-infer/bin/python > "$here/results/inference_requirements.txt"
uv pip freeze --python /opt/verbalizing-subliminal-learning-venv/bin/python > "$here/results/training_requirements.txt"
