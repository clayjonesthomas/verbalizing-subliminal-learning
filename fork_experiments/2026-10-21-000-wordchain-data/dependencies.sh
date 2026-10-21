#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
result="$here/results"
mkdir -p "$result"
mkdir -p "$here/assets" "$here/results"
if [[ ! -d "$here/assets/gepa-legibility/.git" ]]; then
  git clone https://github.com/CDBiddulph/gepa-legibility.git "$here/assets/gepa-legibility"
fi
if [[ $(git -C "$here/assets/gepa-legibility" rev-parse HEAD) != f0558d19afceea51682d8bf1715808f945ee3306 ]]; then
  git -C "$here/assets/gepa-legibility" checkout --detach f0558d19afceea51682d8bf1715808f945ee3306
fi
if [[ ! -x /opt/wordchain-infer/bin/python ]]; then uv venv --python /usr/bin/python3.11 /opt/wordchain-infer; fi
uv pip install --no-cache --python /opt/wordchain-infer/bin/python 'vllm==0.32.0' 'peft==0.19.1' 'dspy @ git+https://github.com/CDBiddulph/dspy.git@f93f96b649b7b794d843f0f4b91309b50e06702c'
uv pip freeze --python /opt/wordchain-infer/bin/python > "$result/inference_requirements.txt"
uv pip freeze --python /opt/verbalizing-subliminal-learning-venv/bin/python > "$result/training_requirements.txt"

uv pip install --no-cache --python /opt/verbalizing-subliminal-learning-venv/bin/python --target /opt/wordchain-merge-libs "transformers==4.57.6"
