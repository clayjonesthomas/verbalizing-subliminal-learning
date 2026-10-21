#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/results" "$here/assets"
trap 'python3 -c "import json,time; from pathlib import Path; Path(\"$here/results/failure.json\").write_text(json.dumps(dict(time=time.time(),error=\"Launch or generation failed; inspect run.log. Pod retained.\")))"' ERR
if [[ ! -x /opt/psychosis-base-infer/bin/python ]]; then uv venv --python /usr/bin/python3.11 /opt/psychosis-base-infer; fi
uv pip install --no-cache --python /opt/psychosis-base-infer/bin/python 'vllm==0.32.0'
uv pip freeze --python /opt/psychosis-base-infer/bin/python > "$here/results/inference_requirements.txt"
tar -cf "$here/assets/environment.tar.tmp" -C / opt/psychosis-base-infer
mv "$here/assets/environment.tar.tmp" "$here/assets/environment.tar"
/opt/psychosis-base-infer/bin/python -u "$here/run.py"
sync -f "$here/results"
